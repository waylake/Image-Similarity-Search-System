import concurrent.futures
import json
import os
from typing import Dict, List, Union, Any
from PIL import Image
import io
import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
from elasticsearch import Elasticsearch, helpers
from fake_headers import Headers
from tqdm import tqdm
import pickle
from dotenv import load_dotenv
from datasketch import MinHash

from .models import BasicInfo
from .logger import logger


class ElasticsearchClient:
    def __init__(self):
        self.host = os.getenv("ELASTICSEARCH_HOST", "elasticsearch")
        self.port = os.getenv("ELASTICSEARCH_PORT", "9200")
        self.client = self._create_client()

    def _create_client(self) -> Elasticsearch:
        try:
            client = Elasticsearch(hosts=f"http://{self.host}:{self.port}")
            logger.info("Elasticsearch client created successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to create Elasticsearch client: {e}")
            raise

    def wait_for_elasticsearch(self, max_retries=10):
        retry_count = 0
        while retry_count < max_retries:
            try:
                health = self.client.cluster.health()
                if health["status"] in ["yellow", "green"]:
                    logger.info("Elasticsearch is ready!")
                    return
            except Exception as e:
                logger.error(f"Error while waiting for Elasticsearch: {e}")
                retry_count += 1
                time.sleep(2)
        raise Exception("Elasticsearch failed to become ready")

    def init_index(self, index_name: str = "images"):
        settings = {
            "mappings": {
                "properties": {
                    "item_name": {"type": "text"},
                    "item_price": {"type": "text"},
                    "item_image": {"type": "keyword"},
                    "item_url": {"type": "keyword"},
                    "shopping_mall_name": {"type": "keyword"},
                    "minhash_vector": {"type": "dense_vector", "dims": 128},
                }
            }
        }

        if self.client.indices.exists(index=index_name):
            self.client.indices.delete(index=index_name)

        self.client.indices.create(index=index_name, body=settings)
        logger.info(f"Index '{index_name}' initialized")


class ImageProcessor:
    @staticmethod
    def process_image(image_url: str, max_size: int = 256) -> Union[None, np.ndarray]:
        error_count = 0
        while error_count < 3:
            try:
                headers = Headers(headers=True).generate()
                response = requests.get(image_url, headers=headers, timeout=10)
                if response.status_code != 200 or not response.content:
                    logger.error(
                        f"Error downloading image: {response.status_code}, {image_url}"
                    )
                    return None

                img = Image.open(io.BytesIO(response.content)).convert("RGB")
                img_np = np.array(img)

                h, w, _ = img_np.shape
                if max(h, w) > max_size:
                    scale = max_size / max(h, w)
                    img_np = cv2.resize(img_np, (int(w * scale), int(h * scale)))

                gray_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
                denoised_img = cv2.fastNlMeansDenoising(
                    gray_img, None, h=10, templateWindowSize=7, searchWindowSize=21
                )
                return denoised_img
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                error_count += 1
        return None

    @staticmethod
    def extract_sift_features(image: np.ndarray) -> Union[None, np.ndarray]:
        sift = cv2.SIFT_create()
        _, descriptors = sift.detectAndCompute(image, None)
        return descriptors

    @staticmethod
    def lsh_hashing(features: np.ndarray, num_perm=128) -> List[int]:
        minhash = MinHash(num_perm=num_perm)
        for feature in features:
            minhash.update(feature.tobytes())
        return list(minhash.hashvalues)


class ItemCollector:
    def __init__(self):
        self.basic_info = BasicInfo()

    def get_item_shopping_mall_name(self, item_url: str) -> Union[str, None]:
        headers = Headers(headers=True).generate()
        response = requests.get(item_url, timeout=30, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        info_table = soup.find("table")
        if info_table:
            rows = info_table.find_all("tr")
            for row in rows:
                if "쇼핑사" in row.text:
                    return row.find_all("td")[1].text.strip()
        return None

    def get_item_info(
        self, items: List[BeautifulSoup], date: str
    ) -> List[Dict[str, str]]:
        results = []
        for item in tqdm(items, desc=f"{date} 상품 정보 수집", total=len(items)):
            try:
                item_name = item.find("div", class_="font-15").text.strip()
                item_price = item.find(
                    "span", class_="strong font-17 c-black"
                ).text.strip()
                item_image = (
                    item.find("img", class_="lazy")
                    .get("data-src")
                    .split("http://thum.buzzni.com/unsafe/320x320/center/smart/")[1]
                )
                item_url = self.basic_info.host + item.find("a").get("href")
                shopping_mall_name = self.get_item_shopping_mall_name(item_url)
                results.append(
                    {
                        "item_name": item_name,
                        "item_price": item_price,
                        "item_image": item_image,
                        "item_url": item_url,
                        "shopping_mall_name": shopping_mall_name,
                    }
                )
            except Exception as e:
                logger.error(f"Error collecting item data: {e}")
        return results

    def get_date_item_info(self, date: str) -> List[Dict[str, str]]:
        retry_count = 0
        while retry_count < 3:
            try:
                headers = Headers(headers=True).generate()
                response = requests.get(
                    f"{self.basic_info.request_url}&date={date}",
                    timeout=30,
                    headers=headers,
                )
                soup = BeautifulSoup(response.text, "html.parser")
                items = soup.find_all("div", class_="timeline-item")
                item_infos = self.get_item_info(items, date)
                with open(f"./data/{date}.json", "w") as f:
                    json.dump(item_infos, f, indent=4, ensure_ascii=False)
                return item_infos
            except Exception as e:
                logger.error(f"Error collecting data for date {date}: {e}")
                retry_count += 1
        return []

    def get_month_item_info(self) -> List[Dict[str, str]]:
        month_item_info = []
        month = self.basic_info.get_30days()
        if not os.path.exists("./data"):
            os.makedirs("./data")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_date = {
                executor.submit(self.get_date_item_info, date): date for date in month
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_date),
                desc="상품 정보 수집",
                total=len(month),
            ):
                date = future_to_date[future]
                try:
                    data = future.result()
                    month_item_info.extend(data)
                    logger.info(f"Data collection completed: {date}")
                except Exception as e:
                    logger.error(f"Error collecting data: {e}")

        return month_item_info


class DataProcessor:
    @staticmethod
    def process_single_item(item_info: Dict[str, str]) -> Union[None, Dict[str, Any]]:
        try:
            image_url = item_info.get("item_image")
            processed_image = ImageProcessor.process_image(image_url)
            if processed_image is None:
                return None

            descriptors = ImageProcessor.extract_sift_features(processed_image)
            if descriptors is None or descriptors.size == 0:
                return None

            minhash = ImageProcessor.lsh_hashing(descriptors)
            item_info["minhash_vector"] = minhash
            return item_info
        except Exception as e:
            logger.error(f"Error processing item information: {e}")
            return None

    @staticmethod
    def extract_sift_features(item_infos: List[Dict[str, str]]) -> List[Dict[str, str]]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(DataProcessor.process_single_item, item_info)
                for item_info in item_infos
            ]

            for idx, future in tqdm(
                enumerate(concurrent.futures.as_completed(futures)),
                desc="SIFT feature 추출",
                total=len(futures),
            ):
                try:
                    result = future.result()
                    if result is not None:
                        item_infos[idx] = result
                except Exception as e:
                    logger.error(f"Error processing item: {e}")

        return [item for item in item_infos if item.get("minhash_vector") is not None]


class ElasticsearchIndexer:
    def __init__(self, es_client: Elasticsearch):
        self.es_client = es_client

    def index_items(
        self, item_infos: List[Dict[str, str]], batch_size: int = 500
    ) -> None:
        batches = []
        for idx, item_info in enumerate(
            tqdm(item_infos, desc="Indexing items", total=len(item_infos))
        ):
            try:
                hashed_features = item_info.get("minhash_vector")
                if hashed_features is not None:
                    action = {
                        "_op_type": "index",
                        "_index": "images",
                        "_source": {
                            "item_name": item_info["item_name"],
                            "item_price": item_info["item_price"],
                            "item_image": item_info["item_image"],
                            "item_url": item_info["item_url"],
                            "shopping_mall_name": item_info["shopping_mall_name"],
                            "minhash_vector": hashed_features,
                        },
                    }
                    batches.append(action)

                if len(batches) >= batch_size or idx == len(item_infos) - 1:
                    helpers.bulk(self.es_client, batches)
                    batches = []
            except Exception as e:
                logger.error(f"Error indexing item: {e}")


def load_environment(env_type):
    try:
        dotenv_path = f"./.env.{env_type}"
        load_dotenv(dotenv_path)
        logger.info(f"Environment loaded: {env_type}")
    except Exception as e:
        logger.error(f"Error loading environment: {e}")


def load_existing_data():
    month = BasicInfo().get_30days()
    item_infos = []
    try:
        for date in month:
            with open(f"./data/{date}.json", "r") as f:
                item_infos.extend(json.load(f))
        logger.info("Existing data loaded")
        return item_infos
    except Exception as e:
        logger.error(f"Error loading existing data: {e}")
        return []


def save_processed_item_infos(processed_item_infos):
    try:
        with open("./data/processed_item_infos.pkl", "wb") as f:
            pickle.dump(processed_item_infos, f)
        logger.info("Processed item information saved")
    except Exception as e:
        logger.error(f"Error saving processed item information: {e}")


def search_similar_images(es: Elasticsearch, image_url: str) -> Dict[str, str]:
    try:
        processed_image = ImageProcessor.process_image(image_url)
        if processed_image is None:
            raise Exception("Error processing image")

        descriptors = ImageProcessor.extract_sift_features(processed_image)
        if descriptors is None or descriptors.size == 0:
            raise Exception("Error extracting SIFT features")

        minhash_vector = ImageProcessor.lsh_hashing(descriptors)

        search_body = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, doc['minhash_vector']) + 1.0",
                        "params": {"query_vector": minhash_vector},
                    },
                }
            }
        }

        search_results = es.search(index="images", body=search_body)
        return search_results
    except Exception as e:
        logger.error(f"Error searching similar images: {e}")
        return {}
