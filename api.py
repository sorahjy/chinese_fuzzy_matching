import bottle, json, sys
from bottle import request, response
from gevent import monkey
from match import Trie

monkey.patch_all()

if len(sys.argv) < 2:
    print('USAGE: python api.py port\neg: python api.py 6657')
    sys.exit()
else:
    port = sys.argv[1]

ENT_LIST = 'data/ent.txt'
app = bottle.Bottle()

trie = Trie(ENT_LIST)
trie.fuzzy_search("诺安基金管理有限公司怎么样")


@app.route('/v1/fuzzy', method=['GET', 'POST'])
def fuzzy_search():
    query = request.params.query
    ret = trie.fuzzy_search(query)
    ret_dic = {'ret': ret}
    return json.dumps(ret_dic, ensure_ascii=False)


@app.route('/v1/exact', method=['GET', 'POST'])
def exact_search():
    query = request.params.query
    ret = trie.exact_search(query)
    ret_dic = {'ret': ret}
    return json.dumps(ret_dic, ensure_ascii=False)


def run_api_server():
    bottle.run(app, host='0.0.0.0', port=port, server="gevent")


if __name__ == '__main__':
    run_api_server()
