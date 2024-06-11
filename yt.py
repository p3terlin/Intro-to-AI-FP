import os
import googleapiclient.discovery

class YouTubeComments:
    def __init__(self, developer_key, api_service_name="youtube", api_version="v3"):
        self.youtube = googleapiclient.discovery.build(
            api_service_name, api_version, developerKey=developer_key)

    def list_comment(self, video_id, page_token='', part='snippet', max_results=100):
        request = self.youtube.commentThreads().list(
            part=part,
            videoId=video_id,
            maxResults=max_results,
            pageToken=page_token
        )
        response = request.execute()

        comments = []
        nextPageToken = response.get('nextPageToken', None)
        items = response.get('items', [])
        for item in items:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append(comment)
        
        return comments, nextPageToken

def main():
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    DEVELOPER_KEY = ""

    youtube_comments = YouTubeComments(DEVELOPER_KEY)
    
    video_id = "oJC8VIDSx_Q"
    nextPageToken = ""
    commentList = []
    while True:
        comments, nextPageToken = youtube_comments.list_comment(video_id, nextPageToken)
        commentList.extend(comments)
        if not nextPageToken:
            break

    # print(commentList)
    
    sorted_commentList = sorted(commentList, key=lambda x: x['likeCount'], reverse=True)

    for comm in sorted_commentList[:10]:
        print(f"Comment: {comm['textOriginal'], comm['likeCount']}")
        print("-----")

if __name__ == "__main__":
    main()
