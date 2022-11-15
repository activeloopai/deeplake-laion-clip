import deeplake
import webdataset as wds
from deeplake.core.sample import Sample

raw_data = 's3://non-hub-datasets-n/laion400m-data/'

ds = deeplake.empty('s3://hub-2.0-datasets-n/laion400m-data-test/')
ds.create_tensor('image', htype='image', sample_compression='jpeg')
ds.create_tensor('caption', htype='text')

files_list = []
for i in range(41408):
    files_list.append(raw_data + str(i).zfill(5) + '.tar')

@deeplake.compute
def process(file, ds_out):
    url = 'pipe:aws s3 cp ' + file + " -"
    with ds:
        wd = wds.WebDataset(url).to_tuple('jpg', 'txt', 'json')
        for img, txt, _ in wd:
            img = Sample(buffer=img, compression='jpeg')
            t = txt.decode()
            if len(img.shape) == 2:
                continue
            ds_out.image.append(img)
            ds_out.caption.append(t)

process().eval(files_list, ds, scheduler='processed', num_workers=6)