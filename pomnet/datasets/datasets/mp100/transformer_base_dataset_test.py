import copy
import math
import os
from abc import ABCMeta, abstractmethod
import json_tricks as json
import numpy as np
import mmcv
import cv2
from matplotlib import pyplot as plt
from mmcv import imshow
from mmcv.image import imwrite
from mmcv.parallel import DataContainer as DC
from mmpose.core.evaluation.top_down_eval import (keypoint_auc, keypoint_epe,
                                                  keypoint_pck_accuracy)
from torch.utils.data import Dataset
from mmpose.datasets import DATASETS
from mmpose.datasets.pipelines import Compose


# @DATASETS.register_module()


def scatter(feat, label, epoch=210):
    plt.ion()
    plt.clf()
    palette = np.array(sns.color_palette('hls', 10))
    ax = plt.subplot(aspect='equal')
    # sc = ax.scatter(feat[:, 0], feat[:, 1], lw=0, s=40, c=palette[label.astype(np.int)])
    for i in range(10):
        plt.plot(feat[label == i, 0], feat[label == i, 1], '.', c=palette[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    ax.axis('tight')
    for i in range(10):
        xtext, ytext = np.median(feat[label == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=18)
        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])

    plt.draw()
    plt.savefig('./benchmark/centerloss_{}.png'.format(epoch))
    plt.pause(0.001)

def show_result(
        img,
        result,
        pck,
        skeleton=None,
        kpt_score_thr=0.3,
        bbox_color='green',
        pose_kpt_color_list=None,
        pose_limb_color_list=None,
        radius=4,
        text_color=(255, 0, 0),
        thickness=1,
        font_scale=0.5,
        win_name='',
        show=False,
        out_dir_lei='',
        wait_time=0,
        mask=None,
        out_file_=None):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        result (list[dict]): The results to draw over `img`
            (bbox_result, pose_result).
        kpt_score_thr (float, optional): Minimum score of keypoints
            to be shown. Default: 0.3.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
            If None, do not draw keypoints.
        pose_limb_color (np.array[Mx3]): Color of M limbs.
            If None, do not draw limbs.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
            Default: 0.
        out_file (str or None): The filename to write the image.
            Default: None.

    Returns:
        Tensor: Visualized img, only if not `show` or `out_file`.
    """

    img = mmcv.imread(img)
    img = img.copy()
    img_h, img_w, _ = img.shape

    bbox_result = []
    pose_result = []
    result[1]['keypoints'] = result[1]['keypoints'] * np.repeat(np.expand_dims(mask, -1), 2, -1)
    for res in result:
        bbox_result.append(res['bbox'])
        pose_result.append(res['keypoints'])

    if len(bbox_result) > 0:
        bboxes = np.vstack(bbox_result)
        # draw bounding boxes
        mmcv.imshow_bboxes(
            img,
            bboxes,
            colors=bbox_color,
            top_k=-1,
            thickness=thickness,
            show=False,
            win_name=win_name,
            wait_time=wait_time,
            out_file=None)

        for person_id, kpts in enumerate(pose_result):
            # draw each point on image
            pose_kpt_color = pose_kpt_color_list[person_id]
            if pose_kpt_color is not None:
                assert len(pose_kpt_color) == len(kpts), (
                    len(pose_kpt_color), len(kpts))
                for kid, kpt in enumerate(kpts):
                    x_coord, y_coord = int(kpt[0]), int(
                        kpt[1])
                    # if kpt_score > kpt_score_thr:
                    if 1 == 1:
                        img_copy = img.copy()
                        r, g, b = pose_kpt_color[kid]
                        cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                                   radius, (int(r), int(g), int(b)), -1)

                        # transparency = max(0, min(1, kpt_score))
                        transparency = 0.5
                        cv2.addWeighted(
                            img_copy,
                            transparency,
                            img,
                            1 - transparency,
                            0,
                            dst=img)
                out_file = '/data/yjliang/code/Category-Agnostic-Pose-Estimation/P2/Pose-for-Everything/work_dirs/token_3_bs16_spilt1_gong_256/fen2_52063/' + out_dir_lei + '/'
                if not os.path.exists(out_file):  # 如果路径不存在
                    os.makedirs(out_file)
                if out_file is not None:
                    imwrite(img, out_file + str(person_id) + '_' + str(pck) + out_file_)

            # draw limbs
            if skeleton is not None and pose_limb_color_list is not None:
                assert len(pose_limb_color_list) == len(skeleton)
                for sk_id, sk in enumerate(skeleton):
                    pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1,
                                                              1]))
                    pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1,
                                                              1]))
                    if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                            and pos1[1] < img_h and pos2[0] > 0
                            and pos2[0] < img_w and pos2[1] > 0
                            and pos2[1] < img_h
                            and kpts[sk[0] - 1, 2] > kpt_score_thr
                            and kpts[sk[1] - 1, 2] > kpt_score_thr):
                        img_copy = img.copy()
                        X = (pos1[0], pos2[0])
                        Y = (pos1[1], pos2[1])
                        mX = np.mean(X)
                        mY = np.mean(Y)
                        length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                        angle = math.degrees(
                            math.atan2(Y[0] - Y[1], X[0] - X[1]))
                        stickwidth = 2
                        polygon = cv2.ellipse2Poly(
                            (int(mX), int(mY)),
                            (int(length / 2), int(stickwidth)), int(angle),
                            0, 360, 1)

                        r, g, b = pose_limb_color_list[sk_id]
                        cv2.fillConvexPoly(img_copy, polygon,
                                           (int(r), int(g), int(b)))
                        transparency = max(
                            0,
                            min(
                                1, 0.5 *
                                   (kpts[sk[0] - 1, 2] + kpts[sk[1] - 1, 2])))
                        cv2.addWeighted(
                            img_copy,
                            transparency,
                            img,
                            1 - transparency,
                            0,
                            dst=img)

    show, wait_time = 0, 1
    if show:
        height, width = img.shape[:2]
        max_ = max(height, width)

        factor = min(1, 800 / max_)
        enlarge = cv2.resize(
            img, (0, 0),
            fx=factor,
            fy=factor,
            interpolation=cv2.INTER_CUBIC)
        # imshow(enlarge, win_name, wait_time)
    out_file = '/data/yjliang/code/Category-Agnostic-Pose-Estimation/P2/Pose-for-Everything/work_dirs/token_3_bs16_spilt1_dri_no_jian_no_pos/out_img/'
    if out_file is not None:
        imwrite(img, out_file + out_file_)

    return img


class TransformerBaseDataset(Dataset, metaclass=ABCMeta):

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 test_mode=False):
        self.image_info = {}
        self.ann_info = {}

        self.annotations_path = ann_file
        self.img_prefix = img_prefix
        self.pipeline = pipeline
        self.test_mode = test_mode

        self.ann_info['image_size'] = np.array(data_cfg['image_size'])
        self.ann_info['heatmap_size'] = np.array(data_cfg['heatmap_size'])
        self.ann_info['num_joints'] = data_cfg['num_joints']

        self.ann_info['flip_pairs'] = None

        self.ann_info['inference_channel'] = data_cfg['inference_channel']
        self.ann_info['num_output_channels'] = data_cfg['num_output_channels']
        self.ann_info['dataset_channel'] = data_cfg['dataset_channel']

        self.db = []
        self.num_shots = 1
        self.paired_samples = []
        self.pipeline = Compose(self.pipeline)

    @abstractmethod
    def _get_db(self):
        """Load dataset."""
        raise NotImplementedError

    @abstractmethod
    def _select_kpt(self, obj, kpt_id):
        """Select kpt."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """Evaluate keypoint results."""
        raise NotImplementedError

    @staticmethod
    def _write_keypoint_results(keypoints, res_file):
        """Write results into a json file."""

        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)

    def _report_metric(self,
                       res_file,
                       metrics,
                       pck_thr=0.2,
                       pckh_thr=0.7,
                       auc_nor=30):
        """Keypoint evaluation.

        Args:
            res_file (str): Json file stored prediction results.
            metrics (str | list[str]): Metric to be performed.
                Options: 'PCK', 'PCKh', 'AUC', 'EPE'.
            pck_thr (float): PCK threshold, default as 0.2.
            pckh_thr (float): PCKh threshold, default as 0.7.
            auc_nor (float): AUC normalization factor, default as 30 pixel.

        Returns:
            List: Evaluation results for evaluation metric.
        """
        info_str = []

        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(self.paired_samples)

        outputs = []
        gts = []
        masks = []
        lei_list = []
        threshold_bbox = []
        threshold_head_box = []
        file_name_list = []
        for pred, pair in zip(preds, self.paired_samples):
            item = self.db[pair[-1]]
            outputs.append(np.array(pred['keypoints'])[:, :-1])
            gts.append(np.array(item['joints_3d'])[:, :-1])
            file_name_list.append(item['image_file'])

            lei_list.append(item['image_file'].split('/')[-2])

            mask_query = ((np.array(item['joints_3d_visible'])[:, 0]) > 0)
            mask_sample = ((np.array(self.db[pair[0]]['joints_3d_visible'])[:, 0]) > 0)
            for id_s in pair[:-1]:
                mask_sample = np.bitwise_and(mask_sample, ((np.array(self.db[id_s]['joints_3d_visible'])[:, 0]) > 0))
            masks.append(np.bitwise_and(mask_query, mask_sample))

            if 'PCK' in metrics:
                bbox = np.array(item['bbox'])
                bbox_thr = np.max(bbox[2:])
                threshold_bbox.append(np.array([bbox_thr, bbox_thr]))
            if 'PCKh' in metrics:
                head_box_thr = item['head_size']
                threshold_head_box.append(
                    np.array([head_box_thr, head_box_thr]))

        if 'PCK' in metrics:
            pck_avg = []
            leis = [lei_list[0]]
            pck_ = []
            for (output, gt, mask, thr_bbox, lei, file_name) in zip(outputs, gts, masks, threshold_bbox, lei_list,
                                                                    file_name_list):
                _, pck, _ = keypoint_pck_accuracy(np.expand_dims(output, 0), np.expand_dims(gt, 0),
                                                  np.expand_dims(mask, 0), pck_thr, np.expand_dims(thr_bbox, 0))

                if lei not in leis:
                    info_str.append(('PCK' + lei, np.mean(pck_)))
                    pck_ = []
                    leis.append(lei)
                pck_.append(pck)
                pck_avg.append(pck)
                output1 = {'keypoints': output, 'bbox': np.zeros(4)}
                gt1 = {'keypoints': gt, 'bbox': np.zeros(4)}
                yan = np.array([255, 0, 0])
                result = [yan for i in range(100)]
                yan = np.array([0, 255, 0])
                gt = [yan for i in range(100)]
                color = [result, gt]
                #if lei == 'squirrel_body':
                #    show_result(file_name, [output1, gt1], pck, mask=mask, pose_kpt_color_list=color,
                #                pose_limb_color_list=color, out_dir_lei=lei,
                #                out_file_=str(pck) + '_' + file_name.split('/')[3])

            info_str.append(('PCK', np.mean(pck_avg)))

        return info_str

    def _merge_obj(self, Xs_list, Xq, idx):
        """ merge Xs_list and Xq.

        :param Xs_list: N-shot samples X
        :param Xq: query X
        :param idx: id of paired_samples
        :return: Xall
        """
        Xall = dict()
        Xall['img_s'] = [Xs['img'] for Xs in Xs_list]
        Xall['target_s'] = [Xs['target'] for Xs in Xs_list]
        Xall['target_weight_s'] = [Xs['target_weight'] for Xs in Xs_list]
        xs_img_metas = [Xs['img_metas'].data for Xs in Xs_list]

        Xall['img_q'] = Xq['img']
        Xall['target_q'] = Xq['target']
        Xall['target_weight_q'] = Xq['target_weight']
        xq_img_metas = Xq['img_metas'].data

        img_metas = dict()
        for key in xq_img_metas.keys():
            img_metas['sample_' + key] = [xs_img_meta[key] for xs_img_meta in xs_img_metas]
            img_metas['query_' + key] = xq_img_metas[key]
        img_metas['bbox_id'] = idx

        Xall['img_metas'] = DC(img_metas, cpu_only=True)

        return Xall

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.paired_samples)

    def __getitem__(self, idx):
        """Get the sample given index."""

        pair_ids = self.paired_samples[idx]
        assert len(pair_ids) == self.num_shots + 1
        sample_id_list = pair_ids[:self.num_shots]
        query_id = pair_ids[-1]

        sample_obj_list = []
        for sample_id in sample_id_list:
            sample_obj = copy.deepcopy(self.db[sample_id])
            sample_obj['ann_info'] = copy.deepcopy(self.ann_info)
            sample_obj_list.append(sample_obj)

        query_obj = copy.deepcopy(self.db[query_id])
        query_obj['ann_info'] = copy.deepcopy(self.ann_info)

        Xs_list = []
        for sample_obj in sample_obj_list:
            Xs = self.pipeline(sample_obj)
            Xs_list.append(Xs)
        Xq = self.pipeline(query_obj)

        Xall = self._merge_obj(Xs_list, Xq, idx)

        return Xall

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        kpts = sorted(kpts, key=lambda x: x[key])
        num = len(kpts)
        for i in range(num - 1, 0, -1):
            if kpts[i][key] == kpts[i - 1][key]:
                del kpts[i]

        return kpts
