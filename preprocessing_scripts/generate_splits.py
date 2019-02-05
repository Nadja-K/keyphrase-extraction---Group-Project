import rethinkdb as r
import random

host = 'cuda2'
port = 28015

with r.connect(host, port, db='keyphrase_extraction') as conn:
    res = r.table('documents').order_by(index=r.desc('id')).has_fields('keyword').filter(r.row['keyword'].count().ge(4)).pluck('id').slice(0, 6200).run(conn)
    id_dict = res.items

    ids = [entry['id'] for entry in id_dict]
    random.shuffle(ids)

    train = ids[:5000]
    dev = ids[5000:5200]
    test = ids[5200:]

    res = {
        'dataset': 'heise',
        'train': train,
        'dev': dev,
        'test': test
    }

    table = r.db('keyphrase_extraction').table('splits')
    table.insert(res).run(conn)