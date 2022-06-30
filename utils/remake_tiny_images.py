from pathlib import Path

def main():

    data_dir = Path('/data/cv/') / 'tiny_images'
    idx_file_path = data_dir / '80mn_cifar_idxs.txt'

    num = 79302017

    with open(idx_file_path, 'r') as idxs:
        cifar_idxs = [int(idx)-1 for idx in idxs]

    # hash table option
    cifar_idxs = set(cifar_idxs)
    in_cifar = lambda x: x in cifar_idxs

    new_count = 0
    with open(data_dir / 'tiny_images.bin', 'rb') as data_file, open(data_dir / 'tiny_images_wo_cifar.bin', 'ab') as new_file:
        i = 0
        while i < num:
            # i should not in cifar
            while in_cifar(i):
                i += 1
            
            data_file.seek(i * 3072)
            new_file.write(data_file.read(3072))
            
            i += 1
            new_count += 1
    print(new_count)

if __name__ == '__main__':

    main()