import csv
import pylast
from pydst.rate_limiters import RateLimited

def setAPI_Connection():
    API_KEY = ""
    API_SECRET = ""
    username = "MarkCutajar"
    password_hash = pylast.md5("")

    network = pylast.LastFMNetwork(api_key=API_KEY, api_secret=API_SECRET,
                                   username=username, password_hash=password_hash)

    return network

@RateLimited(5)  # calls per second
def getTrackTagsLastFM(network, artist_name="Hans Zimmer", track_name="Time", numTags=1):
    track = network.get_track(artist_name, track_name)
    tags = track.get_top_tags(numTags)
    return tags

@RateLimited(5)
def getAlbumTagsLastFM(network, artist_name="Hans Zimmer", album_name="", numTags=1):
    album = network.get_album(artist_name, album_name)
    tags = album.get_top_tags(numTags)
    return tags

if __name__ == "__main__":

    network = setAPI_Connection()

    num_songs = 10
    numTags = 1

    with open("../lastfm/raw_tracks.csv", encoding="utf8") as file:
        reader = csv.reader(file)
        metadata={}
        for rw_idx, row in enumerate(reader):
            if 1<rw_idx <num_songs+2:
                metadata[rw_idx-1] = {'artist': row[5], 'track': row[37], 'album': row[2]}


    track_not_found, tags_present, tags_not_present, album_not_found = 0, 0, 0, 0
    track_and_album_not_present = 0

    print("Number of songs:\t" + str(num_songs))


    for idx in range(1, num_songs+1):

        tags = []
        trackBool, albumBool = False, False

        try:
            tags = getTrackTagsLastFM(network, metadata[idx]['artist'], metadata[idx]['track'], numTags)
            trackBool = True

        except pylast.WSError:
            track_not_found += 1
            try:
                tags = getAlbumTagsLastFM(network, metadata[idx]['artist'], metadata[idx]['album'], numTags)
                albumBool = True

            except pylast.WSError:
                album_not_found += 1


        if len(tags)==0 and (not trackBool and not albumBool):
            track_and_album_not_present += 1
            print(idx)
        elif len(tags)==0 and trackBool:

            try:
                tags = getAlbumTagsLastFM(network, metadata[idx]['artist'], metadata[idx]['album'], numTags)
                albumBool = True
            except pylast.WSError:
                pass
            else:
                if len(tags) == 0:
                    tags_not_present += 1
                    print(idx)
                    print("T {}, A {} np".format(trackBool, albumBool))
                else:
                    tags_present += 1
                    print(idx)
                    print("T {}, A {} p".format(trackBool, albumBool))

        elif len(tags)==0 and albumBool:
            tags_not_present += 1
            print(idx)
            print("T {}, A {}".format(trackBool, albumBool))
        else:
            tags_present += 1
            print(idx)

    print("Track not found:\t" + str(track_not_found))
    print("Album not found:\t" + str(album_not_found))
    print("No song tags:\t" + str(tags_not_present))
    print("Tags present:\t"+ str(tags_present))
    print("Nothing present:\t"+ str(track_and_album_not_present))




# ---------------------------------------------------------------------------------------------------------------------
# Backups
# ---------------------------------------------------------------------------------------------------------------------


    # import requests
#
# response = requests.get('http://ws.audioscrobbler.com/2.0/?method=track.getTags&api_key=cb0650874801a71bc31d503abffe18ee&artist=AC/DC&track=Hells+Bells&user=RJ&format=json')
#
# # 'http://ws.audioscrobbler.com/2.0/?method=track.getTags&api_key=cb0650874801a71bc31d503abffe18ee&artist=Pink+Floyd&track=Coming+Back+To+Life&user=RJ&format=json'
#
# # opening up connection, grabbing the page and close client
#
# print(response.status_code)
# print(response.content)

# ------------------------------------------------------------------------------------------------

# import pylast
#
# API_KEY = "cb0650874801a71bc31d503abffe18ee"
# API_SECRET = "90502a82e6332acb7614b6391bbff9b3"
#
# username = "MarkCutajar"
# password_hash = pylast.md5("lFM&data0")
#
# network = pylast.LastFMNetwork(api_key=API_KEY, api_secret=API_SECRET,
#                                username=username, password_hash=password_hash)
#
# track = network.get_track("Hans Zimmer", "Time")
# tags = track.get_top_tags(100)
# numt = len(tags)
#
# if numt != 0:
#     max_idx = min(10, numt-1)
#     for idx in range(0, max_idx):
#         print(str(tags[idx][0]) + ", " + str(tags[idx][1]))
