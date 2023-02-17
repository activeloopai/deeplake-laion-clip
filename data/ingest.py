import os
import deeplake
import pandas as pd
import tqdm 

def get_parquet_files(path: str) -> list:
    """ get all parquet files from a folder path """
    parquet_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".parquet"):
                parquet_files.append(os.path.join(root, file))
    return parquet_files


if __name__ == "__main__":

    # path to the deeplake dataset
    source_path = "./laion2B-en/"
    path = os.path.join(os.path.dirname(__file__), "laion2b")

    parquet_files = get_parquet_files(source_path)    
    ds = deeplake.empty(path, overwrite=True)

    with ds:
        ds.create_tensor('text', htype="text", chunk_compression='lz4')
        ds.create_tensor('SAMPLE_ID', htype='generic', chunk_compression='lz4')
        ds.create_tensor('URL', htype='text', chunk_compression='lz4') 
        ds.create_tensor('TEXT', htype='text', chunk_compression='lz4')
        ds.create_tensor('HEIGHT', htype='generic', chunk_compression='lz4')
        ds.create_tensor('WIDTH', htype='generic', chunk_compression='lz4')
        ds.create_tensor('LICENSE', htype='text', chunk_compression='lz4')
        ds.create_tensor('NSFW', htype='text', chunk_compression='lz4')
        ds.create_tensor('similarity', htype='generic', chunk_compression='lz4')  

    # read all parquet files
    for parquet_file in tqdm.tqdm(parquet_files):
        df = pd.read_parquet(parquet_file)
        deeplake.ingest_dataframe(df, ds)
        ds.commit(parquet_file)
        ds.summary()