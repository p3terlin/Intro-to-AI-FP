import argparse
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from pytube import YouTube
from moviepy.editor import *
from openai import OpenAI
from faster_whisper import WhisperModel

from bert import BERT, BERTDataset
from preprocess import preprocessing_function

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from yt import YouTubeComments
import os

from deep_translator import GoogleTranslator

OpenAI_DEVELOPER_KEY = ''
GoogleAI_DEVELOPER_KEY = ''

def get_argument():
    # do not modify
    opt = argparse.ArgumentParser()
    opt.add_argument("--yt_url",
                        type=str,
                        help="enter url of youtube video")
    opt.add_argument("--summary",
                        action='store_true',
                        help="summarize the youtube video")
    opt.add_argument("--train",
                        action='store_true',
                        help="train the model")
    opt.add_argument("--start_epoch",
                        type=int,
                        default=0,
                        help="start epoch of training process")
    opt.add_argument("--epochs",
                        type=int,
                        default=1,
                        help="number of epochs in training process")
    opt.add_argument("--eval",
                        action='store_true',
                        help="eval the model")
    opt.add_argument("--N",
                        type=int,
                        default=10,
                        help="show top N likes comments")
    opt.add_argument("--video_path",
                        type=str,
                        default="summary/video.mp4",
                        help="customize the saving path of video")
    opt.add_argument("--txt_path",
                        type=str,
                        default="summary/transcription.txt",
                        help="customize the saving path of transcription")
    
    args = opt.parse_args()
    if (args.eval or args.summary) and not args.yt_url:
        opt.error("--eval and --summary require --yt_url to be specified")
    config = vars(args)
    return config

class VideoSummerizer:
    def __init__(self, video_url, video_path, txt_path):
        self.video_url = video_url # 網址
        self.video_path = video_path # 影片下載後存的位置
        self.txt_path = txt_path # 文字檔存的位置
        self.client = OpenAI(api_key=OpenAI_DEVELOPER_KEY)

    def download_youtube_video(self): # 下載影片
        yt = YouTube(self.video_url)
        os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
        video_stream = yt.streams.filter(file_extension='mp4').first()
        video_stream.download(filename=self.video_path)
        print('YouTube video had been downloaded')
    
    def transcribe_video(self): # 用fast whisper的model把音檔轉成文字檔
        model_size = "large-v3" # tiny, base, small, medium, large, large-v2, large-v3
        # Run on GPU with FP16
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        print('Transcribing...')
        segments, info = model.transcribe(self.video_path, beam_size=5)
        transcription = ""
        transcription_segments = [segment.text for segment in segments]
        transcription = ".".join(transcription_segments)
        with open(self.txt_path, "w") as file:
            file.write(transcription)
        return transcription

    def summarize_text(self, text): # 把文字檔生成摘要
        response = self.client.chat.completions.create(
            # model="gpt-3.5-turbo-0125",
            model="gpt-4o",
            messages=[
                {
                    "role": "system", "content": "You are an helpful assistant who is good at organizing videos. Please answer in traditional chinese.",
                    "role": "user", "content": f"Answer the following questions with traditional chinese (繁體中文): \n1. Please first summarize the following text in 200 to 300 words\n 2. Create 5 self-question-and-answer based on the text:\n\n{text}",
                }
            ],
            temperature=0.7,
        )
        summary = response.choices[0].message.content
        return summary

class BERTTrainer:
    def __init__(self, model_type, df_train, N, bert_config, start_epoch, epochs):
        self.model_type = model_type
        self.df_train = df_train
        self.N = N
        self.bert_config = bert_config
        self.start_epoch = start_epoch
        self.epochs = epochs

        self.model = self.load_model()
        self.train_dataloader = self.prepare_data()

    def load_model(self):
        model = BERT('distilbert-base-uncased', config=self.bert_config)
        return model
    
    def collate_fn(self, data):
        sequences, labels = zip(*data)
        sequences, labels = list(sequences), list(labels)
        sequences = self.model.tokenizer(sequences, padding=True, truncation=True,max_length=512, return_tensors="pt")
        return sequences,torch.tensor(labels)

    def prepare_data(self):
        train_data = BERTDataset(self.df_train)
        train_dataloader = DataLoader(train_data, batch_size=self.bert_config['batch_size'], collate_fn=self.collate_fn)
        return train_dataloader

    def train(self):
        '''
        total_loss: the accumulated loss of training
        labels: the correct labels set for the test set
        pred: the predicted labels set for the test set
        '''
        device = self.bert_config['device']
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.bert_config['lr'])
        loss_fn = nn.CrossEntropyLoss().to(device)

        pretrained = "./models/BERT_{}.pt".format(self.start_epoch-1)
        if os.path.exists(pretrained):
            print("=> loading checkpoint '{}'".format(pretrained))
            self.model.model = torch.load(pretrained)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(pretrained, self.start_epoch-1))
        else:
            self.start_epoch = 0 
            print('=> no pretrained model, start training from the beginning.')

        for epoch in range(self.start_epoch, self.epochs):
            total_loss = 0.0
            self.model.train()  # Set the model to training mode

            print("=> Start training epoch " + str(epoch))
            # training stage
            for batch in tqdm(self.train_dataloader):
                inputs, labels = tuple(t.to(device) for t in batch)
                optimizer.zero_grad()  # Clear gradients
                pred = self.model.forward(inputs)
                loss = loss_fn(pred, labels)
                loss.backward()  # Backpropagation
                optimizer.step()  # Update parameters
                total_loss += loss.item()  # Update total loss

            file_name = '.models' + self.model_type + '_' + str(epoch)
            self.model.model.epoch = epoch
            print('=> Saving classifier to ' + file_name)
            torch.save(self.model.model, file_name+".pt")

class SentimentAnalyzer:
    def __init__(self, model_path, config, tokenizer):
        self.model = torch.load(model_path, map_location=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.tokenizer = tokenizer
        self.device = config['device']
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode

    def predict(self, text):
        # Preprocess the input text
        preprocessed_text = preprocessing_function(text)
        # Tokenize and prepare the input for the model
        inputs = self.tokenizer(preprocessed_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        label_mapping = {0: 'negative', 1: 'positive'}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            outputs = nn.Softmax(dim=1)(outputs)

        # Get the predicted label and its probability
        probabilities = outputs.detach().cpu().numpy()
        predicted_label_idx = np.argmax(probabilities, axis=1)[0]
        predicted_label = label_mapping[predicted_label_idx]
        predicted_probability = probabilities[0][predicted_label_idx]
    
        return predicted_label, predicted_probability

def get_youtube_video_id(url):
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return None

def main():
    # get argument
    model_type = "BERT"
    preprocessed = True
    N = 2 # we only use bi-gram in this assignment, but you can try different N
    label_mapping = {'negative': 0, 'positive': 1}
    
    bert_config = {
        'batch_size': 8,
        'lr': 2e-5,
        'device': torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    }

    config = get_argument()
    if config['yt_url'] is not None:
        video_url = config['yt_url']
        video_id = get_youtube_video_id(video_url)

    if config['summary']:
        video_path = config['video_path']
        txt_path = config['txt_path']
        summerizer  = VideoSummerizer(video_url=video_url, video_path=video_path, txt_path=txt_path)
        summerizer.download_youtube_video()
        transcription_text = summerizer.transcribe_video()
        summary = summerizer.summarize_text(transcription_text)
        print(summary)

    if config['train']:
        start_epoch = config['start_epoch']
        epochs = config['epochs']

        # read and prepare data
        df_train = pd.read_csv('./data/IMDB_train.csv',index_col=None, header=0)
        df_train['review'] = df_train['review'].apply(preprocessing_function)
        df_train['sentiment'] = df_train['sentiment'].map(label_mapping)

        trainer = BERTTrainer(model_type, df_train, N, bert_config, start_epoch, epochs)
        trainer.train()
    
    if config['eval']:
        top_N = config['N']

        model_path = './models/BERT_5.pt'  # Path to the saved model
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        analyzer = SentimentAnalyzer(model_path, bert_config, tokenizer)

        # os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

        youtube_comments = YouTubeComments(developer_key=GoogleAI_DEVELOPER_KEY)
        
        # video_id = "oJC8VIDSx_Q"
        nextPageToken = ""
        commentList = []
        pos_cnt = 0
        neg_cnt = 0
        while True:
            comments, nextPageToken = youtube_comments.list_comment(video_id, nextPageToken)
            commentList.extend(comments)
            if not nextPageToken:
                break

        sorted_commentList = sorted(commentList, key=lambda x: x['likeCount'], reverse=True)
        print("Top {} Likes Comments\n\n".format(top_N))
        # Iterate through the sorted comments
        for i, comm in enumerate(sorted_commentList):
            # Translate the comment text
            text = comm['textOriginal']
            translated = GoogleTranslator(source='auto', target='en').translate(text=text)
            
            if translated is not None:
                # Analyze sentiment of the translated text
                sentiment, probabilities = analyzer.predict(translated)
            
            # Increment counters based on sentiment
            if sentiment == "positive":
                pos_cnt += 1
            else:
                neg_cnt += 1
            
            # Print details for the top N comments
            if i < top_N:
                print(text)
                # print(translated)
                print(f"Sentiment: {sentiment}, Probabilities: {probabilities}, Likes: {comm['likeCount']}")
                print("---------------------------------------")
            
            # Stop processing further comments if the like count is zero
            if comm['likeCount'] <= 0:
                break

        # Calculate and print the percentage of positive and negative comments
        total_comments = pos_cnt + neg_cnt
        print("\nResults: ({} available comments)".format(total_comments))
        if total_comments > 0:
            pos_percentage = pos_cnt / total_comments * 100
            neg_percentage = neg_cnt / total_comments * 100
            print("Positive comments: {:.2f}%   Negative comments: {:.2f}%\n".format(pos_percentage, neg_percentage))
        else:
            print("No comments to analyze.")

if __name__ == '__main__':
    main()