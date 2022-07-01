from pathlib import Path

def main():

    data_dir = Path('/data/cv/') / 'tiny_images'
    idx_file_path = data_dir / '80mn_cifar_idxs.txt'

    num = 79302017

    with open(idx_file_path, 'r') as idxs:
        cifar_idxs = [int(idx)-1 for idx in idxs]

    # hash table option
    cifar_idxs = set(cifar_idxs)
    # print(len(cifar_idxs))
    in_cifar = lambda x: x in cifar_idxs

    data_file = open(data_dir / 'tiny_images.bin', 'rb')
    w_c_count, wo_c_count, i = 0, 0, 0
    
    with open(data_dir / 'tiny_images_w_cifar.bin', 'ab') as w_c_f, open(data_dir / 'tiny_images_wo_cifar.bin', 'ab') as wo_c_f:
        while i < num:
            data_file.seek(i * 3072)
            if in_cifar(i):
                w_c_f.write(data_file.read(3072))
                w_c_count += 1
            else:
                # wo_c_f.write(data_file.read(3072))
                wo_c_count += 1
            
            i += 1
    
    data_file.close()
    print(w_c_count, wo_c_count)

if __name__ == '__main__':

    main()