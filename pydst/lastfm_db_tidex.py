import sys
import sqlite3


def die_with_usage():
    """ HELP MENU """
    print('demo_tags_db.py')
    print('  by T. Bertin-Mahieux (2011) tb2332@columbia.edu')
    print('')
    print('Shows how to use the SQLite database made from tags')
    print('in the Last.fm dataset.')
    print('')
    print('USAGE:')
    print('  ./demo_tags_db.py <tags.db>')
    sys.exit(0)


if __name__ == '__main__':

    # open connection
    conn = sqlite3.connect('lastfm/lastfm_tags.db')

    tid = 'TRCCOFQ128F4285A9E'
    print('We get all tags (with value) for track: {}'.format(tid))
    print(' ')

    sql = "SELECT tags.tag, tid_tag.val FROM tags, tid_tag, tids \
            WHERE tags.ROWID = tid_tag.tag AND tid_tag.tid=tids.ROWID AND tids.tid='%s'" % tid

    # sql = "SELECT tid_tag.val FROM tid_tag"
    res = conn.execute(sql)
    data = res.fetchall()
    print(data)
