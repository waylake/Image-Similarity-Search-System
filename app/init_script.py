import argparse
from .utils import (
    ElasticsearchClient,
    ItemCollector,
    DataProcessor,
    ElasticsearchIndexer,
    load_environment,
    load_existing_data,
    save_processed_item_infos,
)
from .logger import logger


def main(update_data: bool, env_type: str):
    load_environment(env_type)
    logger.info(f"Environment: {env_type}")
    logger.info(f"Update data: {update_data}")

    es_client = ElasticsearchClient()
    es_client.wait_for_elasticsearch()
    es_client.init_index()

    try:
        if update_data:
            item_collector = ItemCollector()
            item_infos = item_collector.get_month_item_info()
        else:
            item_infos = load_existing_data()

        processed_item_infos = DataProcessor.extract_sift_features(item_infos)

        indexer = ElasticsearchIndexer(es_client.client)
        indexer.index_items(processed_item_infos)

        save_processed_item_infos(processed_item_infos)
    except Exception as e:
        logger.error(f"An error occurred during initialization: {e}")
        return

    logger.info("Initialization completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update_data", action="store_true", help="Update data from source"
    )
    parser.add_argument(
        "--env_type",
        type=str,
        default="localhost",
        choices=["localhost", "docker"],
        help="Environment type",
    )
    args = parser.parse_args()
    main(args.update_data, args.env_type)
