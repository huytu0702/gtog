#!/usr/bin/env python3
"""Test Cosmos DB connection"""
import os

# Set env vars
os.environ['COSMOS_ENDPOINT'] = 'https://cosmos-emulator:8081'
os.environ['COSMOS_KEY'] = 'C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+ZLw=='

from azure.cosmos import CosmosClient, PartitionKey

client = CosmosClient(
    url=os.environ['COSMOS_ENDPOINT'],
    credential=os.environ['COSMOS_KEY'],
    connection_verify=False,
)

# List databases
print("Listing databases...")
databases = list(client.list_databases())
print(f"Found {len(databases)} databases")
for db in databases:
    print(f"  - {db['id']}")

# Create or get database
print("\nCreating database 'gtog'...")
try:
    database = client.create_database_if_not_exists(id='gtog')
    print("Database created or exists")
except Exception as e:
    print(f"Error: {e}")
    database = client.get_database_client('gtog')

# List containers
print("\nListing containers...")
try:
    containers = list(database.list_containers())
    print(f"Found {len(containers)} containers")
    for container in containers:
        print(f"  - {container['id']}")
except Exception as e:
    print(f"Error: {e}")

# Create or get container
print("\nCreating container 'graphrag'...")
try:
    container = database.create_container_if_not_exists(
        id='graphrag',
        partition_key=PartitionKey(path='/id'),
        offer_throughput=400
    )
    print("Container created or exists")
except Exception as e:
    print(f"Error: {e}")
    container = database.get_container_client('graphrag')

print("\nDone!")
