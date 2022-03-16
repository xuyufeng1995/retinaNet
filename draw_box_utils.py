import collections
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import numpy as np
from pathlib import Path

STANDARD_COLORS = [
    'red', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',  # AliceBlue
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def filter_low_thresh(boxes, scores, classes, category_index, thresh, box_to_display_str_map, box_to_color_map):
    for i in range(boxes.shape[0]):
        if scores[i] > thresh:
            box = tuple(boxes[i].tolist())  # numpy -> list -> tuple
            if classes[i] in category_index.keys():
                class_name = category_index[classes[i]]
            else:
                class_name = 'N/A'
            display_str = str(class_name)
            display_str = '{}: {}%'.format(display_str, int(100 * scores[i]))
            box_to_display_str_map[box].append(display_str)
            box_to_color_map[box] = STANDARD_COLORS[
                classes[i] % len(STANDARD_COLORS)]
        else:
            break  # 网络输出概率已经排序过，当遇到一个不满足后面的肯定不满足


def draw_text(draw, box_to_display_str_map, box, left, right, top, bottom, color):
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in box_to_display_str_map[box]]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in box_to_display_str_map[box][::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)], fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill='black',
                  font=font)
        text_bottom -= text_height - 2 * margin


def draw_box(image, boxes, classes, scores, category_index, thresh=0.5, line_thickness=8):
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)

    filter_low_thresh(boxes, scores, classes, category_index, thresh, box_to_display_str_map, box_to_color_map)

    # Draw all boxes onto image.
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    for box, color in box_to_color_map.items():
        xmin, ymin, xmax, ymax = box
        (left, right, top, bottom) = (xmin * 1, xmax * 1,
                                      ymin * 1, ymax * 1)
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=line_thickness, fill=color)
        draw_text(draw, box_to_display_str_map, box, left, right, top, bottom, color)


# def plot_labels(labels, names=(), save_dir=Path(''), loggers=None):
#     # plot dataset labels
#     print('Plotting labels... ')
#     c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
#     nc = int(c.max() + 1)  # number of classes
#     colors = color_list()
#     x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height'])
#
#     # seaborn correlogram
#     sns.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
#     plt.savefig(save_dir / 'labels_correlogram.jpg', dpi=200)
#     plt.close()
#
#     # matplotlib labels
#     matplotlib.use('svg')  # faster
#     ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
#     ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
#     ax[0].set_ylabel('instances')
#     if 0 < len(names) < 30:
#         ax[0].set_xticks(range(len(names)))
#         ax[0].set_xticklabels(names, rotation=90, fontsize=10)
#     else:
#         ax[0].set_xlabel('classes')
#     sns.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)
#     sns.histplot(x, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)
#
#     # rectangles
#     labels[:, 1:3] = 0.5  # center
#     labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000
#     img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
#     for cls, *box in labels[:1000]:
#         ImageDraw.Draw(img).rectangle(box, width=1, outline=colors[int(cls) % 10])  # plot
#     ax[1].imshow(img)
#     ax[1].axis('off')
#
#     for a in [0, 1, 2, 3]:
#         for s in ['top', 'right', 'left', 'bottom']:
#             ax[a].spines[s].set_visible(False)
#
#     plt.savefig(save_dir / 'labels.jpg', dpi=200)
#     matplotlib.use('Agg')
#     plt.close()
#
#     # loggers
#     for k, v in loggers.items() or {}:
#         if k == 'wandb' and v:
#             v.log({"Labels": [v.Image(str(x), caption=x.name) for x in save_dir.glob('*labels*.jpg')]}, commit=False)
