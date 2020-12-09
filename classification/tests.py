import matplotlib.pyplot as plt
from data.datasets import danbooruFacesCrops

def test_data_load_danbooruFacesCrops():
    dataset = danbooruFacesCrops(split='val')
    print(dataset)
    print(dataset.data_dic_pd.head())
    print(dataset.class_names_pd.head())
    print(dataset.data_dic_pd.columns)
    print(dataset.class_names_pd.columns)

    print(dataset.data_dic.shape)
    print(dataset.class_names.shape)
    print(dataset.data_dic[0, :])
    print(dataset.class_names[0, :])
    
    # test methods
    print(len(dataset))
    print(dataset.no_classes())
    dataset.stats()
    
    # test data get item
    #print(dataset[0])
    image, label = dataset[0]
    fig = plt.figure()
    plt.imshow(image)
    plt.show()
    print(label)

    print('Finished successfully')

def main():
    test_data_load_danbooruFacesCrops()

main()
