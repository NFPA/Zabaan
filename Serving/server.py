import os
import json
import tornado.ioloop
import tornado.web
import tornado.httpserver
import tornado.web
import tensorflow as tf

import client
from datetime import datetime, timezone
from io import StringIO
from db_model import feedback_collection


class PredictHandler(tornado.web.RequestHandler):

    def post(self, *args, **kwargs):
        text = self.get_argument("text")
        # predict = client.translate_multiway(client.stub, client.args.model_name, self.get_argument("text"), client.tokenizer,
        #                           timeout=client.args.timeout, tgt_lang=self.get_argument('tgt'))
        predict = client.translate(client.stub, client.args.model_name, self.get_argument("text"), client.tokenizer,
                                  timeout=client.args.timeout)
        data = {
            'input': text,
            'output': predict
        }
        self.write(json.dumps({'data': data}))


class UploadHandler(tornado.web.RequestHandler):
    def post(self):
        f = self.request.files['file'][0]
        f = f['body'].decode('utf-8')
        t = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

        if not os.path.exists(os.getcwd() + '/upload'):
            os.system('mkdir -p %s' % (os.getcwd() + '/upload'))
        # x = open(os.getcwd() + '/upload/%s.txt' % t, 'w+')
        # x.write(f)
        # x.close()

        f = f.split('\n')
        result = client.translate_file(client.stub, client.args.model_name, f, client.tokenizer,
                                       timeout=client.args.timeout)
        content = {
            'raw': '\n'.join(f),
            'target': '\n'.join([v[1:-1] for v in result])
        }

        self.set_header('Content-Type', 'application/json; charset=UTF-8')
        self.write({'ok': True, 'content': content})


class CodesHandler(tornado.web.RequestHandler):
    def get(self, *args, **kwargs):
        code_list = client.get_code_list()
        self.write({'ok': True, 'content': code_list})


class SentenceHandler(tornado.web.RequestHandler):
    def get(self):
        result = client.shuffle_file(self.get_argument('code'))
        self.write({'ok': True, 'content': result})

    def post(self, *args, **kwargs):
        translation = \
        client.translate(client.stub, client.args.model_name, self.get_argument('source'), client.tokenizer,
                         timeout=client.args.timeout)[0]
        self.write({'ok': True, 'content': translation})


class FeedBackHandler(tornado.web.RequestHandler):
    def post(self, *args, **kwargs):
        code = self.get_argument('code')
        raw = self.get_argument('raw')
        payload = json.loads(self.get_argument('payload'))

        feedback_collection.insert({
            'code': code,
            'raw': raw,
            'payload': payload
        })

        self.write({'ok': True})


def make_app():
    root = os.getcwd()
    print(root)
    return tornado.web.Application([
        (r"/upload", UploadHandler),
        (r"/predict", PredictHandler),
        (r"/codes", CodesHandler),
        (r"/sentence", SentenceHandler),
        (r"/feedback", FeedBackHandler),
        (r'/(.*)',
         tornado.web.StaticFileHandler,
         {'path': root + '/template/', 'default_filename': 'index.html'})
    ], debug=True)


if __name__ == "__main__":
    app = make_app()
    app.listen(8080)
    print("listen start.....")
    tornado.ioloop.IOLoop.current().start()
