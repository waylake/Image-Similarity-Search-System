from fastapi import FastAPI, HTTPException, Query, Depends
from elasticsearch import Elasticsearch
from .utils import ElasticsearchClient, search_similar_images
from .logger import logger

app = FastAPI()


def get_es_client():
    return ElasticsearchClient().client


@app.get("/items/search/")
async def search_item(
    image_url: str = Query(..., description="Input image url"),
    es: Elasticsearch = Depends(get_es_client),
):
    try:
        search_results = search_similar_images(es, image_url)
        return search_results
    except Exception as e:
        logger.error(f"Error in search_item: {e}")
        raise HTTPException(status_code=400, detail=str(e))
