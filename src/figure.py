import matplotlib.pyplot as plt

def plot():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(156, 45, marker='.', color='blue')
    plt.plot(171, 62, marker='.', color='blue')
    plt.plot(180, 72, marker='.', color='blue')
    plt.plot(153, 42, marker='.', color='blue')
    plt.plot(170, 60, marker='.', color='blue')
    plt.plot(158, 53, marker='.', color='blue')
    plt.plot(173, 59, marker='.', color='blue')
    plt.plot(176, 67, marker='.', color='blue')
    plt.plot(181, 78, marker='.', color='blue')
    plt.plot(166, 65, marker='.', color='blue')
    plt.plot(167, 70, marker='.', color='blue')
    plt.plot(163, 59, marker='.', color='blue')
    plt.plot(161, 62, marker='.', color='blue')
    plt.plot(178, 74, marker='.', color='blue')
    plt.plot(172, 66, marker='.', color='blue')
    plt.plot(173, 70, marker='.', color='blue')
    plt.plot(162, 62, marker='.', color='blue')
    plt.plot(155, 52, marker='.', color='blue')
    plt.plot(153, 55, marker='.', color='blue')
    plt.plot(163, 58, marker='.', color='blue')
    plt.plot(163, 58, marker='.', color='blue')
    plt.plot(161, 54, marker='.', color='blue')
    plt.plot(165, 57, marker='.', color='blue')
    plt.title('height, weight')
    plt.xlabel('height(cm)')
    plt.ylabel('weight(kg)')

    mean = [167, 61]
    point1 = [182, 77]
    point2 = [164,68]

    ax.annotate('', xy=point1, xytext=mean,
                arrowprops=dict(shrink=0, width=1, headwidth=8, headlength=10, connectionstyle='arc3',
                                facecolor='red', edgecolor='red'))
    
    ax.annotate('', xy=point2, xytext=mean,
                arrowprops=dict(shrink=0, width=1, headwidth=8, headlength=10, connectionstyle='arc3',
                                facecolor='red', edgecolor='red'))
    plt.savefig('figure.png')