from train import train_multi_modal_rec_model


if __name__ == '__main__':
    dev_set_file = './MovieLens-dataset-with-poster-main/dev_set.csv'
    posters_dir = './MovieLens-dataset-with-poster-main/poster/'
    train_multi_modal_rec_model(dev_set_file, posters_dir)