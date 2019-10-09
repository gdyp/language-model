#! -*- coding: utf-8 -*-
import json
import numpy as np
import tornado.web
from tornado.options import define, options
from tornado.httpserver import HTTPServer

from test import get_hidden_state
#
# logger = logging.basicConfig(level='INFO')


class SentenceEmbeddingHandler(tornado.web.RequestHandler):
    def post(self):
        data = self.request.body.decode()
        sentence = json.loads(data)['sentence']
        ans_dict = dict(code=2100, message="", data={})
        try:
            if sentence:
                ans_dict = get_hidden_state(sentence)
                if isinstance(ans_dict, np.ndarray):
                    ans_dict = ans_dict.tolist()
        except Exception as e:
            ans_dict = dict(code=4000, message=str(e), data={})
        ans_str = json.dumps(ans_dict, ensure_ascii=False)
        self.write(ans_str)


application = tornado.web.Application([
    (r"/sentence_embedding/", SentenceEmbeddingHandler),
])

if __name__ == "__main__":
    myserver = HTTPServer(application)
    myserver.bind(1368)
    myserver.start(num_processes=1)
    print('sentence paraphraser server is running....')
    tornado.ioloop.IOLoop.current().start()
