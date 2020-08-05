import convenience
import matplotlib.pyplot as plt
import collections

if __name__ == "__main__":
    citances = convenience.get_all_citances()

    print("Number of citances", len(citances))

    facet_counts_multi = collections.defaultdict(int)
    facet_counts_single = collections.defaultdict(int)
    for cit in citances:
        facet_counts_multi[" & ".join(sorted(cit["Discourse Facet"]))] += 1
        for facet in cit["Discourse Facet"]:
            facet_counts_single[facet] += 1

    # for k, v in facet_counts_multi.items():
    #     print(k, v)

    # print("----_")

    # for k, v in facet_counts_single.items():
    #     print(k, v)

    # print("------")

    # print(sum(facet_counts_single.values()), sum(facet_counts_multi.values()))

    labels_multi = facet_counts_multi.keys()
    labels_single = facet_counts_single.keys()

    sizes_multi = facet_counts_multi.values()
    sizes_single = facet_counts_single.values()

    fig1, ax1 = plt.subplots()
    wedges, texts = ax1.pie(sizes_multi, startangle=90)
    ax1.axis('equal')
    ax1.set_title = "Multi"
    ax1.set_position([-0.2,0, 1, 1])
    ax1.legend(wedges, labels_multi,
          loc="center left",
          bbox_to_anchor=(0.84, 0, 0.5, 1), 
          prop={'size': 20},
          )
    # plt.show()
    plt.savefig("./multi_facet_counts.png", bbox_inches="tight")

    fig1, ax1 = plt.subplots()
    wedges, texts = ax1.pie(sizes_single, startangle=90)
    ax1.axis('equal')
    ax1.set_title = "Single"
    ax1.set_position([-0.2,0, 1, 1])
    ax1.legend(wedges, labels_single,
          loc="center left",
          bbox_to_anchor=(0.84, 0, 0.5, 1), 
          prop={'size': 20},
          )
    # plt.show()
    plt.savefig("./single_facet_counts.png", bbox_inches="tight")