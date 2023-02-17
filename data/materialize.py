import os
import deeplake
import traceback

@deeplake.compute
def fetch_image(sample_in, sample_out):
    return

    try:
        url = sample_in['URL'].data()['value']
        if sample_in['NSFW'].data()['value'] != "UNLIKELY":
            return

        img = deeplake.read(url, verify=True, compression='jpeg')

        sample_out.append({
            "image": img, 
            "text": sample_in['TEXT'].data()['value'],
        })
    except Exception as e:
        print(e)


if __name__ == "__main__":

    path = os.path.join(os.path.dirname(__file__), "laion2b")
    path_dest = os.path.join(os.path.dirname(__file__), "laion-image")

    ds = deeplake.load(path, read_only=True)
    ds.summary()

    ds_new = deeplake.empty(path_dest, overwrite=True)
    ds_new.create_tensor('image', htype="image", sample_compression='jpeg')
    ds_new.create_tensor('text', htype="text", chunk_compression='lz4')
    
    step = 1_000_000
    for i in range(0, len(ds), step):
        fetch_image().eval(ds[:1], ds_new, scheduler='processed', num_workers=2)
        ds_new.commit()