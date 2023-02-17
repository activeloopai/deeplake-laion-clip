import os
import deeplake

@deeplake.compute
def fetch_image(sample_in, sample_out):
    try:
        url = sample_in['URL'].data()['value']
        img = deeplake.read(url, verify=True, compression='jpeg')

        if sample_in['NSFW'].data()['value'] != "UNLIKELY":
            return

        sample_out.append({
            "image": img, 
            "text": sample_in['TEXT'].data()['value'],
        })
    except Exception as e:
        print(e)


if __name__ == "__main__":

    path = os.path.join(os.path.dirname(__file__), "laion2b")
    path_dest = os.path.join(os.path.dirname(__file__), "image")

    ds = deeplake.load(path, read_only=True)
    # ds.delete_tensor('text')
    ds.summary()

    ds_new = deeplake.empty(path_dest, overwrite=True)
    ds_new.create_tensor('image', htype="image", sample_compression='jpeg')
    ds_new.create_tensor('text', htype="text", chunk_compression='lz4')
    
    step = 1_000_000 # 1_000_000
    for i in range(0, len(ds), step):
        fetch_image().eval(ds[i:step+i], ds_new, scheduler='threaded', num_workers=1280)
        ds_new.commit()
        break