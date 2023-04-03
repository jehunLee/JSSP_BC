
# learning figure ##################################################################################
def save_fig(valids, losses, train_folder, model_name):
    # plt show ################################################################################
    plt.clf()

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('repeat_n')
    ax1.set_ylabel('loss')
    line1 = ax1.plot(losses, color='green', label='train loss')

    ax2 = ax1.twinx()
    ax2.set_ylabel('makespan')
    line2 = ax2.plot(valids, color='deeppink', label='valid makespan')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.title(model_name)
    plt.savefig(train_folder + model_name + '.png')
