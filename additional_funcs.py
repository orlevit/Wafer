import cv2
import random
import itertools
import numpy as np
from tqdm import tqdm
from scipy.ndimage import label
from matplotlib import pyplot as plt
from skimage.measure import regionprops
from config import IF_SAVE_IMAGES, IMAGES_SAVE_LOC, RANDOM_STATE, TEST_IND, \
    MIN_SEG_AREA_ALLOWED, COMBINATION_SIZE_LIMIT, POLY_ORDER, SCORE_STRENGTH_OF_INK_PIXELS, TRAIN_IND, \
    WAFER_SAMPLE_SIZE, MIDDLE_RIGHT_WAFER_SIZE_RANGE, MIDDLE_LEFT_WAFER_SIZE_RANGE, \
    RIGHT_WAFER_SIZE_RANGE


def get_component_size(area):
    inc_seg_size = 0
    while True:
        adjusted_min_seg_area_allowed = MIN_SEG_AREA_ALLOWED + inc_seg_size
        number_of_segments = sum(adjusted_min_seg_area_allowed <= area)
        inc_seg_size += 1

        if number_of_segments <= COMBINATION_SIZE_LIMIT:
            return adjusted_min_seg_area_allowed


def find_best_fit(score_fit_pixels_vec, score_fit_slender_vec, n_pix_vec):
    score_work = np.exp(0.1 * np.array(score_fit_pixels_vec)) * score_fit_slender_vec
    score_work = score_work * n_pix_vec
    err_work = 1 / score_work

    return err_work


def draw_on_image(X, Y, IsGoodDie, IsScratchDie):
    img = np.zeros((max(Y) + 1, max(X) + 1))
    rgb = np.zeros((*img.shape, 3))
    xy_scratch = []

    for k in range(len(X)):
        if IsGoodDie.iloc[k] == True:
            if IsScratchDie.iloc[k] == True:
                gv = 100  # Ink Die
                rgb[Y.iloc[k], X.iloc[k], :] = [1, 1, 0]
            else:
                gv = 20  # Good Die
                rgb[Y.iloc[k], X.iloc[k], :] = [0, 0.5, 0]  # Green
        else:
            if IsScratchDie.iloc[k] == True:
                gv = 250  # Bad scratch Die
                rgb[Y.iloc[k], X.iloc[k], :] = [0, 0, 1]
                xy_scratch.append([X.iloc[k], Y.iloc[k]])
            else:
                gv = 200  # Bad simple Die
                rgb[Y.iloc[k], X.iloc[k], :] = [1, 0, 0]
        img[Y.iloc[k], X.iloc[k]] = gv

    return img, rgb, xy_scratch


def fit_pixels_to_scratch(xy_scratch, poly_order):
    x_min = np.min(xy_scratch[:, 0])
    x_max = np.max(xy_scratch[:, 0])
    y_min = np.min(xy_scratch[:, 1])
    y_max = np.max(xy_scratch[:, 1])

    dx = x_max - x_min
    dy = y_max - y_min

    X = xy_scratch[:, 0]
    Y = xy_scratch[:, 1]

    if dx > dy:
        ii = np.where(X == np.min(X))
        xy1 = [X[ii[0][0]], Y[ii[0][0]]]
        ii = np.where(X == np.max(X))
        xy2 = [X[ii[0][0]], Y[ii[0][0]]]

    else:
        ii = np.where(Y == np.min(Y))
        xy1 = [X[ii[0][0]], Y[ii[0][0]]]
        ii = np.where(Y == np.max(Y))
        xy2 = [X[ii[0][0]], Y[ii[0][0]]]
        aa = 1
    slope = (xy2[1] - xy1[1]) / (xy2[0] - xy1[0])
    alpha_rad = np.arctan(slope)

    # Rotate to the horizontal position
    x0 = np.mean(X)
    y0 = np.mean(Y)
    X_tag = X - x0
    Y_tag = Y - y0

    s = np.sin(alpha_rad)
    c = np.cos(alpha_rad)

    x_rot = X_tag * c + Y_tag * s
    y_rot = -X_tag * s + Y_tag * c

    # Fit
    p = np.polyfit(x_rot, y_rot, poly_order)
    x1 = np.min(x_rot)
    x2 = np.max(x_rot)

    # X - Orig Resolution
    X_fit, Y_fit, Y_fit_rot, dxy_fit = calc_Y_fit(p, x_rot, alpha_rad, x0, y0)
    # Calculate Error
    err_L1 = np.mean(np.abs(Y - Y_fit))
    err_L2 = np.mean((Y - Y_fit) ** 2)

    # X - High Resolution (HR)
    xx = np.arange(x1, x2, 0.1)
    X_fit_HR, Y_fit_HR, Y_fit_HR_rot, dxy_fit_HR = calc_Y_fit(p, xx, alpha_rad, x0, y0)

    XY_fit = np.column_stack((X_fit, Y_fit))
    XY_fit_HR = np.column_stack((X_fit_HR, Y_fit_HR))

    err = {'L1': err_L1, 'L2': err_L2}

    return XY_fit, XY_fit_HR, err, dxy_fit_HR


def calc_Y_fit(p, xx, alpha_rad, x0, y0):
    yy = np.polyval(p, xx)
    Y_fit_rot = yy

    # For output
    dx = np.max(xx) - np.min(xx)
    dy = np.max(Y_fit_rot) - np.min(Y_fit_rot)
    dxy_fit = [dx, dy]

    # Rotate back to the original position
    s = np.sin(-alpha_rad)
    c = np.cos(-alpha_rad)

    xx1 = xx * c + yy * s
    yy1 = -xx * s + yy * c

    X_fit = xx1 + x0
    Y_fit = yy1 + y0

    return X_fit, Y_fit, Y_fit_rot, dxy_fit


def save_image(i, WaferName_1, rgb, XY_fit_HR, poly_order, score_strength_of_ink_pixels):
    if IF_SAVE_IMAGES:
        plt.imshow(rgb)
        plt.scatter(XY_fit_HR[:, 0], XY_fit_HR[:, 1], c='cyan', s=1**-10)
        fig_name = f"{i}_{WaferName_1}_poly_{poly_order}_strength_{score_strength_of_ink_pixels}.png"
        plt.savefig(f"{IMAGES_SAVE_LOC}/{fig_name}")
        plt.clf()


def wafer_predicted_scratch(row, XY_fit_HR_int):
    row_x = row['DieX']
    row_y = row['DieY']
    for x_arr, y_arr in XY_fit_HR_int:
        if (row_x == x_arr) and (row_y == y_arr):
            return True

    return False


def metrics_calc(WaferName_df, XY_fit_HR, results_dict):
    XY_fit_HR_int = np.unique(XY_fit_HR.astype(int), axis=0)
    WaferName_df['predicted'] = WaferName_df.apply(lambda x: wafer_predicted_scratch(x, XY_fit_HR_int), axis=1)

    WaferName_scratch_df = WaferName_df[(WaferName_df['IsScratchDie'] == True)]
    WaferName_non_scratch_df = WaferName_df[(WaferName_df['IsScratchDie'] == False)]

    # Recall - on all the true good die scratch
    good_scratch_die_df = WaferName_scratch_df[WaferName_scratch_df['IsGoodDie'] == True]
    if len(good_scratch_die_df):
        good_scratch_die_count = len(good_scratch_die_df[good_scratch_die_df['predicted'] == True])
        recall_scratch_good_ink = good_scratch_die_count / len(good_scratch_die_df)
        results_dict['recall_scratch_good_ink'].append(recall_scratch_good_ink)

    fn = len(WaferName_scratch_df[WaferName_scratch_df.predicted == False])
    tp = len(WaferName_scratch_df[WaferName_scratch_df.predicted == True])
    fp = len(WaferName_non_scratch_df[WaferName_non_scratch_df.predicted == True])

    scratch_recall = None
    scratch_precision = None
    if tp + fp:
        scratch_precision = tp / (tp + fp)
        results_dict['scratch_precision'].append(scratch_precision)

    if tp + fn:
        scratch_recall = tp / (tp + fn)
        results_dict['scratch_recall'].append(scratch_recall)

    if scratch_recall is not None and scratch_precision is not None and (scratch_precision + scratch_recall):
        scratch_f1 = 2 * scratch_precision * scratch_recall / (scratch_precision + scratch_recall)
        results_dict['scratch_f1'].append(scratch_f1)


def filter_wafer_by_size(row, a1, b1, a2, b2):
    wafer_size = row['wafer_size']
    if wafer_size < MIDDLE_LEFT_WAFER_SIZE_RANGE:
        return False
    elif MIDDLE_LEFT_WAFER_SIZE_RANGE <= wafer_size <= MIDDLE_RIGHT_WAFER_SIZE_RANGE:
        slop = a1
        intercept = b1
    else:
        slop = a2
        intercept = b2

    if slop * wafer_size + intercept <= row['bad_percentage']:
        return False
    else:
        return True


def select_train_data(data_df, train_ind):
    agg_df = data_df.groupby(['WaferName']). \
        agg({'DieX': 'max', 'IsGoodDie': 'sum', 'WaferName': 'count'}). \
        rename(columns={'DieX': 'wafer_size', 'IsGoodDie': 'sum', 'WaferName': 'count'})
    agg_df['bad_percentage'] = (agg_df['count'] - agg_df['sum']) / agg_df['count']

    x1 = [MIDDLE_LEFT_WAFER_SIZE_RANGE, MIDDLE_RIGHT_WAFER_SIZE_RANGE]
    y1 = [0.15, 0.118]
    x2 = [MIDDLE_RIGHT_WAFER_SIZE_RANGE, RIGHT_WAFER_SIZE_RANGE]
    y2 = [0.118, 0.105]

    a1, b1 = np.polyfit(x1, y1, deg=1)
    a2, b2 = np.polyfit(x2, y2, deg=1)

    agg_df['take_wafer'] = agg_df.apply(lambda row: filter_wafer_by_size(row, a1, b1, a2, b2), axis=1)
    threshold_wafers = agg_df[agg_df['take_wafer'] == True].index.to_list()

    if train_ind == TEST_IND:
        return data_df[data_df['WaferName'].isin(threshold_wafers)]

    random.seed(RANDOM_STATE)
    selected_wafers = random.sample(threshold_wafers, k=WAFER_SAMPLE_SIZE)
    return data_df[data_df['WaferName'].isin(selected_wafers)]


def calc_df(data_df, poly_order, score_strength_of_ink_pixels, train_pred_ind):
    WaferName_cell = data_df['WaferName']
    WaferName_unique = WaferName_cell.unique()
    results_dict = {'scratch_f1': [], 'scratch_precision': [], 'scratch_recall': [], 'recall_scratch_good_ink': []}

    print(f'Check hyper-parameters: {poly_order}, {score_strength_of_ink_pixels}')
    for i in tqdm(range(len(WaferName_unique))):
        WaferName_1 = WaferName_unique[i]
        ii = WaferName_cell.str.match(WaferName_1)

        # -- Construct Image --
        WaferName_df = data_df.loc[ii]
        X = WaferName_df['DieX']
        Y = WaferName_df['DieY']
        IsGoodDie = WaferName_df['IsGoodDie']

        if train_pred_ind == TRAIN_IND:
            IsScratchDie = WaferName_df['IsScratchDie']
        else:
            WaferName_df['IsScratchDie'] = False
            IsScratchDie = WaferName_df['IsScratchDie']

        img, rgb, xy_scratch = draw_on_image(X, Y, IsGoodDie, IsScratchDie)

        # --- Black White (BW) images---
        bw_full_wafer = img > 0
        bw_bad_die = img >= 200

        # Remove edge dies
        bw_full_wafer1 = cv2.erode(bw_full_wafer.astype(np.uint8), np.ones((3, 3)))
        bw_bad_die_with_no_edge = bw_bad_die & bw_full_wafer1
        bw = bw_bad_die_with_no_edge

        # Connected components (Original)
        clr, n_clr = label(bw, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        STATS = regionprops(clr)
        area = np.array([stat.area for stat in STATS])

        # Image with big segments only
        bw1 = np.zeros_like(bw)

        min_seg_area_allowed_adjusted = get_component_size(area)

        for k in range(n_clr):
            if min_seg_area_allowed_adjusted <= area[k]:
                xy = STATS[k].coords
                for j in range(xy.shape[0]):
                    bw1[xy[j, 0], xy[j, 1]] = 1

        # Connected components (big segments only)
        clr1, n_clr1 = label(bw1, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        STATS1 = regionprops(clr1)

        k_vec = []
        j_vec = []
        n_pix_vec = []
        score_fit_pixels_vec = []
        score_fit_slender_vec = []
        slend_vec = []
        res_struct = []

        for k in range(n_clr1):
            P = list(itertools.combinations(range(n_clr1), k + 1))
            no_of_combinations = len(P)
            for j in range(no_of_combinations):
                indices_of_segmentCombination = P[j]
                xy_scratch_candidates = []
                no_of_segs_in_combinations = len(indices_of_segmentCombination)
                for j1 in range(no_of_segs_in_combinations):
                    indices = indices_of_segmentCombination[j1]  # The indices are clr-numbers of "clr1"
                    xy1 = STATS1[indices].coords
                    xy_scratch_candidates.extend(xy1)

                xy_scratch_candidates = np.array(xy_scratch_candidates).reshape(-1, 2)[:, [1, 0]]

                # --- Fit ---
                [XY_fit, XY_fit_HR, err, dxy_fit_HR] = fit_pixels_to_scratch(xy_scratch_candidates, poly_order)
                dxy = dxy_fit_HR

                # -- Draw current pixels under evaluation --
                # plt.figure(4)
                # plt.imshow(clr1, cmap='gray')
                # plt.plot(xy_scratch_candidates[:, 0], xy_scratch_candidates[:, 1], '*y')
                # plt.plot(xy_scratch_candidates[:, 0], xy_scratch_candidates[:, 1], '+m')

                # Find Scratch and Ink Pixels
                bw_detected = np.zeros_like(bw_bad_die)
                [ny, nx] = bw_bad_die.shape
                for k1 in range(XY_fit_HR.shape[0]):
                    xy = np.round(XY_fit_HR[k1, :]).astype(int)
                    if 0 <= xy[0] < nx and 0 <= xy[1] < ny:
                        bw_detected[xy[1], xy[0]] = 1

                # In case "bw_detected" is larger than original
                [ny, nx] = bw_bad_die.shape
                bw_detected = bw_detected[:ny, :nx]
                ii_bad, _ = np.where(bw_detected & bw_bad_die)
                ii_ink, _ = np.where(bw_detected & ~bw_bad_die)
                n_pix = len(xy_scratch_candidates)  # No. of pixels

                k_vec.append(k)
                j_vec.append(j)
                n_pix_vec.append(n_pix)

                # -- Score --
                score_fit_pixels = len(ii_bad) + len(ii_ink) * score_strength_of_ink_pixels
                slend = dxy[1] / dxy[0]  # slenderness
                score_fit_slender = max(-2 * slend + 1, 0.01)

                score_fit_pixels_vec.append(score_fit_pixels)
                slend_vec.append(slend)
                score_fit_slender_vec.append(score_fit_slender)

                res_struct.append({
                    'k': k,
                    'j': j,
                    'ind': indices_of_segmentCombination,
                    'err': err,
                    'xy_scratch_real': xy_scratch_candidates,
                    'XY_fit': XY_fit,
                    'XY_fit_HR': XY_fit_HR
                })

        if not score_fit_pixels_vec:
            continue

        # Find best fit
        err_work = find_best_fit(score_fit_pixels_vec, score_fit_slender_vec, n_pix_vec)

        best_fit_index = np.argmin(err_work)
        XY_fit_HR = res_struct[best_fit_index]['XY_fit_HR']

        if train_pred_ind == TRAIN_IND:
            save_image(i, WaferName_1, rgb, XY_fit_HR, poly_order, score_strength_of_ink_pixels)
            metrics_calc(WaferName_df, XY_fit_HR, results_dict)
        else:
            pred_scratch(data_df, WaferName_df, XY_fit_HR)

    return results_dict


def pred_scratch(data_df, WaferName_df, XY_fit_HR):
    XY_fit_HR_int = np.unique(XY_fit_HR.astype(int), axis=0)
    WaferName_df['IsScratchDie'] = WaferName_df.apply(lambda x: wafer_predicted_scratch(x, XY_fit_HR_int), axis=1)
    WaferName_scrach_pred_index = WaferName_df[WaferName_df['IsScratchDie'] == True].index.to_list()
    data_df.loc[WaferName_scrach_pred_index, 'IsScratchDie'] = True


def hyperparameters_search(train_df):
    search_dict = {}
    for poly_order in POLY_ORDER:
        for score_strength_of_ink_pixels in SCORE_STRENGTH_OF_INK_PIXELS:
            # tic = time.time()
            results_dict = calc_df(train_df, poly_order, score_strength_of_ink_pixels, TRAIN_IND)
            # results_dict['time'].append((time.time() - tic) / 60)

            mean_results_dict = {}
            for k, v in results_dict.items():
                mean_results_dict[k] = np.mean(v)
            search_dict[(poly_order, score_strength_of_ink_pixels)] = mean_results_dict

    best_fit_values = sorted(search_dict.items(),
                             key=lambda x: (
                             x[1]['scratch_f1'], x[1]['recall_scratch_good_ink'], -x[0][0], x[0][1]))[-1]

    best_poly_order = best_fit_values[0][0]
    best_score_strength_of_ink_pixels = best_fit_values[0][1]

    print(f'Best fit values:\npoly_order:{best_poly_order}\n'
          f'score_strength_of_ink_pixels:{best_score_strength_of_ink_pixels}\nResults:{best_fit_values[1]}')

    return best_poly_order, best_score_strength_of_ink_pixels


def print_stat(df, df_selected, test_ind):
    total_wafers = len(df["WaferName"].unique())
    count_wafers = len(df_selected["WaferName"].unique())
    print(f'Total Wafers count: {total_wafers}. Wafers above threshold: {count_wafers}. '
          f'Percentage taken: {round(count_wafers/total_wafers, 2)}%')