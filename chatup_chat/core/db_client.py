
from typing import List
import requests
from chatup_chat.config import config


class DatabaseApiClient:
    def __init__(self):
        self.config = config
        self.db_base_url = self.config.database.url_api_version

    def _gen_url(self, route):
        return f"{self.db_base_url}/{route}"

    def _make_request(self, method, route, **kwargs):
        response = method(self._gen_url(route), **kwargs)
        response.raise_for_status()
        return response.json()

    def get_shop_prompt(self, shop_id):
        return self._make_request(requests.get, f"shops/{shop_id}/prompt")["prompt"]

    def get_shop_contact_info(self, shop_id):
        return self._make_request(requests.get, f"shops/{shop_id}")["contact_email"]

    def get_closest_shop_doc(self, embedding: List[float], shop_id: int):
        data = {
            "query_embedding": embedding
        }
        docs = self._make_request(requests.post, f"shops/{shop_id}/closest-doc", json=data)["documents"]
        context_doc = ""
        for doc in docs[:5]:
            context_doc += doc
        return context_doc
