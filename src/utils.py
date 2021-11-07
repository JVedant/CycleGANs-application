import matplotlib.pyplot as plt

# A method to generate a sample image on which we can see the training efficiency visually
def generate_images(model, test_image, epoch):
    predicted = model(test_image)

    plt.figure(figsize=(12,12))
    display_list = [test_image[0], predicted[0]]
    titles = ['Test Image', 'Predicted Image']

    #for i in range(len(display_list)):
    #    plt.subplot(1, 2, i+1)
    #    plt.title(title[i])
    #    # getting the pixel values between [0, 1] to plot it.
    #    plt.imsave(f"../Logs/output_images/{epoch}", display_list[i] * 0.5 + 0.5)
    #    plt.axis('off')

    fig = plt.figure()

    ax1 = fig.add_subplot(2,1,1)
    #ax1.title(titles[0])
    ax1.imshow(display_list[0] * 0.5 + 0.5)

    ax2 = fig.add_subplot(2,1,2)
    #ax2.title(titles[1])
    ax2.imshow(display_list[1] * 0.5 + 0.5)

    # Save the full figure...
    fig.savefig(f"../Logs/output_images/{epoch}")


    #plt.show()