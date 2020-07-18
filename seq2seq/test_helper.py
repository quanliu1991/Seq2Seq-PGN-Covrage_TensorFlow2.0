#!/usr/bin/env python3
# -*- coding=utf-8 -*-
# @author   : Yang Heng
# @date     : 2019-12-15
import numpy as np
import time
import tensorflow as tf
from tqdm import tqdm
from rouge import Rouge

from seq2seq.beam_search_helper import beam_search
from seq2seq.model_seq2seq_lq import Seq2Seq
from utils import data, config
from utils.common_utils import init_logger
from utils.wv_utils import WVTool
from matplotlib import pyplot as plt

# 日志配置
logger = init_logger()

def batch_decode(batcher, model: Seq2Seq, vocab, sep: str = ' ') :
    start = time.time()
    counter = 0
    batch = batcher.next_batch()

    rouge_score = []
    rouge = Rouge()
    rouge_high = []
    predict_bad = []
    while batch is not None:
        # Run beam search to get best Hypothesis
        best_summary = beam_search(batch,model,vocab)

        # Extract the output ids from the hypothesis and convert back to words
        output_ids = [int(t) for t in best_summary.tokens[1:]]
        decoded_words = data.outputids2words(output_ids, vocab,
                                             (batch.art_oovs[0] if config.pointer_gen else None))

        # Remove the [STOP] token from decoded_words, if necessary
        try:
            fst_stop_idx = decoded_words.index(data.STOP_DECODING)
            decoded_words = decoded_words[:fst_stop_idx]
        except ValueError:
            decoded_words = decoded_words

        original_articles = batch.original_articles[0]
        original_abstract_sents = ' '.join(batch.original_abstracts_sents[0])

        decoded_words = " ".join(decoded_words)


        rouge_1 = rouge.get_scores(decoded_words, original_abstract_sents)[0]["rouge-1"]["f"]
        if rouge_1<0.4:
            print("\n原文：", original_articles)
            print("权要：", original_abstract_sents)
            print("预测：", decoded_words)
            print("ROUGE:",rouge_1)
            rouge_high.append(rouge_1)
        else:
            predict={"原文:":original_articles,"权要：":original_abstract_sents,"预测：":decoded_words}
            predict_bad.append(predict)
        rouge_score.append(rouge_1)
        # write_for_rouge(original_abstract_sents, decoded_words, counter,
        #                 self._rouge_ref_dir, self._rouge_dec_dir)
        counter += 1
        if counter % 1000 == 0:
            print('%d example in %d sec' % (counter, time.time() - start))
            start = time.time()

        batch = batcher.next_batch()
    print('rouge_high_unm=', len(rouge_high))
    for bad in predict_bad:
        print("\n原文：", bad["原文:"])
        print("权要：", bad["权要："])
        print("预测：",  bad["预测："])

    rouge_mean = np.array(rouge_score).mean()
    print("Decoder has finished reading dataset for single_pass.")
    print("rouge_1:", rouge_mean)
    print("rouge_score:",rouge_score)


    # 参数依次为list,抬头,X轴标签,Y轴标签,XY轴的范围
    def draw_hist(myList, Title, Xlabel, Ylabel, Xmin, Xmax, Ymin, Ymax):
        plt.hist(myList, 10)
        plt.xlabel(Xlabel)
        plt.xlim(Xmin, Xmax)
        plt.ylabel(Ylabel)
        plt.ylim(Ymin, Ymax)
        plt.title(Title)
        plt.show()

    draw_hist(rouge_score, 'ROUGEList', 'score', 'number', 0, 250, 0.0, 1)  # 直方图展示


    # results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
    # rouge_log(results_dict, self._decode_dir)


    # 遍历数据
    # with tqdm(enumerate(dataset.take(steps)), unit='batch', desc='测试进度') as pbar:
    #     for step, batch in pbar:
    #         if test_mode == 'greedy':
    #             results += batch_greedy_decode(batch, model, vocab, sep)
    #         elif test_mode == 'beam':
    #             results += batch_beam_decode(batch, model, vocab,  sep)
    #         else:
    #             raise Exception('未知的推理模式!')
    #         pbar.set_postfix({'size': '{:d}'.format(batch_size)})
    #         pbar.update(1)
    # # 返回结果
    # return results

if __name__=="__main__":
    rouge_score=[0.540540535836377, 0.17647058325259532, 0.35555555164444447, 0.25641025363247866, 0.4833333295833334, 0.5882352892733564, 0.4374999954882812, 0.6037735803061588, 0.11428571108571438, 0.5423728763688597, 0.49999999545000007, 0.4705882303114187, 0.3235294071972319, 0.6666666620222222, 0.39393939042699727, 0.6111111061265434, 0.36363635867768596, 0.6388888842746914, 0.3749999954882812, 0.434782603705104, 0.7407407357475996, 0.6666666618666668, 0.5999999952, 0.2857142831107355, 0.557377044213921, 0.999999995, 0.999999995, 0.33333332932098775, 0.5882352895501731, 0.47058823061899274, 0.37499999501953135, 0.3846153812064431, 0.3188405748372191, 0.32258064079084287, 0.6896551677051131, 0.2499999955555556, 0.38095237636054424, 0.5454545414876033, 0.6976744138236886, 0.5142857099755102, 0.30434782108695657, 0.5624999957031251, 0.4545454495867769, 0.6363636314049588, 0.641509429263083, 0.36363635867768596, 0.9387755052061642, 0.4666666620222223, 0.5581395301027584, 0.6190476146031747, 0.7999999950222222, 0.4634146295062463, 0.23999999500800012, 0.6976744136722554, 0.5599999952000001, 0.5384615342011835, 0.8484848435261708, 0.4999999950500001, 0.7999999950080001, 0.3749999953125, 0.4109588996960031, 0.434782603705104, 0.4999999950000001, 0.3749999962500001, 0.4081632605581008, 0.5599999953920002, 0.7272727223140496, 0.3529411719031142, 0.6428571382653062, 0.5333333283333335, 0.46666666186666667, 0.8301886743325028, 0.5245901589357701, 0.7199999950720002, 0.41666666180555556, 0.7499999950781251, 0.41935483472944857, 0.4117647017301038, 0.6857142807183674, 0.622950814673475, 0.5999999952, 0.4999999950500001, 0.6133333285333334, 0.5797101400126025, 0.5142857093877552, 0.5714285667120181, 0.3018867893912424, 0.1999999952000001, 0.6363636314049588, 0.26229507707605487, 0.535714280739796, 0.4999999953719009, 0.6976744139318551, 0.5853658490184416, 0.41666666236979166, 0.7692307644970415, 0.4999999950000001, 0.5333333285333334, 0.3703703663039172, 0.39999999573964495, 0.679999995392, 0.6666666620385676, 0.49999999538580253, 0.18279569726384554, 0.30769230316568047, 0.6428571379591838, 0.4102564062064431, 0.3043478220321362, 0.3478260819659736, 0.7317073120761451, 0.699999995, 0.46153845701183444, 0.3999999955555556, 0.5483870921071801, 0.6190476144557824, 0.285714281122449, 0.42105262670360116, 0.11111110786694109, 0.21428571078514744, 0.42553191007695795, 0.3736263705880933, 0.28888888611111113, 0.4999999950000001, 0.7391304297920607, 0.33333332847222225, 0.5666666617166667, 0.6153846106508877, 0.5238095190929706, 0.999999995, 0.2173913012003781, 0.5999999953555556, 0.652173908383743, 0.7499999950781251, 0.5333333285333334, 0.5423728772536627, 0.2912621326760298, 0.285714281122449, 0.24999999531250006, 0.41509433611961555, 0.4912280653616498, 0.461538457433432, 0.8571428522448981, 0.4814814769204391, 0.6956521691493385, 0.44897958708871305, 0.7428571378938778, 0.9333333283555556, 0.46153845665680476, 0.624999995378125, 0.5116279025851812, 0.4615384565680473, 0.6111111063580248, 0.5161290276795006, 0.7333333285333336, 0.5925925879286695, 0.518518513580247, 0.7719298195998769, 0.6818181769111571, 0.7843137205382545, 0.5624999951757813, 0.5333333283555556, 0.49180327375436717, 0.4074074024691358, 0.5454545404958678, 0.5818181768198348, 0.6829268242950625, 0.5185185135253774, 0.39999999504132233, 0.26086956030245756, 0.5714285664399092, 0.5806451563371489, 0.5614035037857804, 0.46666666186666667, 0.559999995008, 0.6086956474102081, 0.2580645111342353, 0.49999999531250006, 0.5999999952, 0.47058823029604, 0.49056603337842647, 0.888888883888889, 0.6315789424238228, 0.5555555508024692, 0.8124999950781252, 0.7619047570068028, 0.37499999500000003, 0.6666666622222223, 0.4285714236734694, 0.42696628865042296, 0.8148148098216735, 0.4615384568047337, 0.8571428521428571, 0.45454544954545456, 0.7999999951125, 0.4999999950500001, 0.5517241334126041, 0.6666666618916438, 0.5581395302109249, 0.37037036543209884, 0.3999999957289257, 0.5714285664399092, 0.3999999964734694, 0.46666666168888893, 0.999999995, 0.5517241334126041, 0.34042552709823454, 0.24489795478550605, 0.8461538411834321, 0.7777777730246914, 0.879999995008, 0.6060606011753903, 0.5714285664285715, 0.4166666625347223, 0.27777777500000006, 0.7999999952000001, 0.3404255279130829, 0.36363635867768596, 0.7142857093877552, 0.5853658486853064, 0.5416666623697917, 0.2857142819642857, 0.7499999950781251, 0.7777777728395062, 0.6285714237714286, 0.35294117148788934, 0.46808510176550483, 0.7499999950347224, 0.874999995, 0.5882352895501731, 0.5945945895982471, 0.3255813911303408, 0.6060606017217631, 0.22222221739369008, 0.8108108058436816, 0.39999999513388435, 0.8421052581717452, 0.6666666616666668, 0.8627450930565167, 0.6666666622222223, 0.999999995, 0.3589743539776463, 0.4727272677685951, 0.6285714235755103, 0.5882352895501731, 0.6666666617555557, 0.6984126935651298, 0.6363636315289257, 0.7058823483737025, 0.7179487129783038, 0.3636363586363637, 0.5945945897735574, 0.6111111061111112, 0.6181818132231407, 0.24999999580000004, 0.3076923029585799, 0.2727272697520661, 0.5454545408264463, 0.5999999953555556, 0.7142857095153062, 0.7499999953125, 0.5833333283333335, 0.43636363295206615, 0.6666666618055556, 0.6666666616699541, 0.6666666618666668, 0.44444443961591223, 0.42424241935720847, 0.47619047129251707, 0.8095238047052155, 0.8979591786922116, 0.49999999501953135, 0.35714285278061225, 0.9142857092897959, 0.66666666207483, 0.7878787829201102, 0.6666666620385676, 0.6666666616666668, 0.3703703657064472, 0.6285714237714286, 0.7142857093877552, 0.4782608645746692, 0.4666666620222223, 0.7586206846611178, 0.43478260415879016, 0.4150943346244215, 0.2999999950500001, 0.4406779624820454, 0.7777777728395062, 0.5581395308599243, 0.409090904927686, 0.4583333292013889, 0.5263157844875347, 0.5882352892733564, 0.8636363587293389, 0.38095237605442184, 0.734693872553103, 0.2916666640299479, 0.39999999500800004, 0.5161290273048909, 0.46511627420227153, 0.7999999950000002, 0.7586206846611178, 0.5454545406198348, 0.7142857092857143, 0.6874999954882813, 0.4406779614248779, 0.6333333283333334, 0.3291139206729691, 0.4363636314049587, 0.43181817806818185, 0.6666666619501135, 0.4615384577251808, 0.43749999507812504, 0.5217391255198489, 0.42424241935720847, 0.7692307642307693, 0.648648643652301, 0.8571428521449397, 0.5882352892733564, 0.37931034011890613, 0.5161290273881375, 0.46511627485127094, 0.2325581374797188, 0.43243242787436087, 0.6999999952000001, 0.5217391254442345, 0.3333333284222223, 0.6285714236734695, 0.6111111061111112, 0.1678321660658223, 0.5806451571696151, 0.31999999520000005, 0.2424242374288339, 0.6999999950500001, 0.5283018818796725, 0.22222221722222232, 0.6315789423822715, 0.6428571378826532, 0.2903225757752342, 0.7407407358024692, 0.3888888841358025, 0.4242424196143251, 0.3333333283333334, 0.1100917420789496, 0.491228065189289, 0.8387096725494276, 0.5161290275130074, 0.31818181421487607, 0.3199999950080001, 0.611111106867284, 0.6874999950195313, 0.24999999625000005, 0.6428571378826532, 0.4999999951757813, 0.4615384572781066, 0.5769230721893491, 0.4827586158382878, 0.6666666616666668, 0.4230769180769231, 0.47619047129251707, 0.3529411716262976, 0.5999999952, 0.2727272683884298, 0.3043478211247638, 0.46153845662064436, 0.5666666616666667, 0.5217391260396976, 0.2727272683884298, 0.48888888497777777, 0.7199999950079999, 0.666666661781451, 0.5789473639196676, 0.35714285216836733, 0.6923076874260355, 0.24999999561250005, 0.9090909041322315, 0.4324324284879475, 0.9166666617013889, 0.6666666617687076, 0.7199999950720002, 0.6896551674375743, 0.38095237599773246, 0.999999995, 0.7272727224380167, 0.8837209253001623, 0.11267605412418175, 0.5777777728000001, 0.5333333286055556, 0.23076922721893492, 0.5999999950000001, 0.3809523770861678, 0.627450975717032, 0.42622950335931203, 0.3703703653772291, 0.5624999950195313, 0.5945945897735574, 0.2448979559350271, 0.46808510176550483, 0.4999999950255103, 0.7027026977063551, 0.48648648149013884, 0.6206896504637337, 0.4242424197979798, 0.32876712054043916, 0.29999999625, 0.5882352891349482, 0.2857142820158268, 0.5405405361577795, 0.6666666617120182, 0.6451612853277836, 0.78260869073724, 0.3636363595107438, 0.5405405359824691, 0.8387096724245579, 0.5882352897404844, 0.7857142807142857, 0.2978723356994116, 0.2456140319729148, 0.29885057019421324, 0.5476190432568028, 0.6896551678953627, 0.3749999955555556, 0.6666666617687076, 0.37681159048939306, 0.7333333284222223, 0.4999999950500001, 0.5333333286888889, 0.6666666622222223, 0.5185185138545954, 0.9142857092897959, 0.6923076877810652, 0.5714285664399092, 0.6530612197417743, 0.666666661701389, 0.592592587654321, 0.5999999952, 0.45161289864724247, 0.6060606011019284, 0.4166666617013889, 0.2539682489795919, 0.38461537964497045, 0.5882352891349482, 0.79999999505, 0.6363636315289257, 0.5070422496012696, 0.5376344042317032, 0.45833332833333335, 0.7428571378938778, 0.45161289823100936, 0.6666666617687076, 0.6399999950080001, 0.4444444394513032, 0.7999999950888891, 0.4888888840888889, 0.3076923041420118, 0.39393938997245176, 0.8571428522448981, 0.45161289864724247, 0.46874999517578125, 0.8275862019024971, 0.49180327497984416, 0.5882352892733564, 0.9629629579698218, 0.8421052581717452, 0.36363635867768596, 0.5454545408264463, 0.6249999953125001, 0.7999999950080001, 0.5999999950888889, 0.34782608380907376, 0.4888888840098766, 0.33333332958333334, 0.43243242861577796, 0.5909090867458678, 0.7083333284722222, 0.7878787828833793, 0.21052631313019393, 0.35555555111111115, 0.11538461050295878, 0.5925925879286695, 0.4999999950413224, 0.5853658495181441, 0.639999995032, 0.5185185138545954, 0.5263157844875347, 0.3157894688088643, 0.49180327466810003, 0.8837209252352624, 0.7450980342329875, 0.25581395004597085, 0.3999999954938776, 0.5098039165859286, 0.17391304089792062, 0.699999995, 0.23529411307958487, 0.5714285665306124, 0.6799999951280001, 0.3999999953125, 0.7857142808163265, 0.41379309873959574, 0.5263157851523547, 0.5714285664399092, 0.7567567517896275, 0.879999995008, 0.2999999950500001, 0.7346938725697626, 0.44210525948808876, 0.533333328888889, 0.7142857093877552, 0.6896551675624257, 0.7083333283680555, 0.7647058773529413, 0.37735848706301184, 0.7619047569160999, 0.7499999951125002, 0.3333333283333334, 0.6956521689224953, 0.47619047120181407, 0.4545454499173554, 0.518518513580247, 0.374999995703125, 0.491228065189289, 0.4117647010380623, 0.5263157845983379, 0.3214285680612245, 0.7857142808163265, 0.7659574420823904, 0.749999995, 0.5384615339349113, 0.49999999524691374, 0.7368421003878117, 0.22222221777777784, 0.3181818140185951, 0.6470588188062285, 0.5925925881481482, 0.22222221777777784, 0.6086956472589792, 0.5116279021957817, 0.704225347232692, 0.372093018864251, 0.5454545406198348, 0.8749999950195313, 0.6666666616666668, 0.5957446761249435, 0.7804877999524094, 0.6285714236734695, 0.4166666617013889, 0.6315789428670361, 0.6153846104142012, 0.6666666618055556, 0.4210526272576177, 0.7179487129783038, 0.799999995072, 0.7407407357407407, 0.4285714236734694, 0.5714285668367348, 0.5405405356318481, 0.49999999531250006, 0.7586206847086802, 0.49999999545000007, 0.6746987905646683, 0.39999999505, 0.7777777727932099, 0.5128205083760684, 0.7741935435171696, 0.578947363767313, 0.6451612857440167, 0.39999999520000007, 0.16326530397334446, 0.49999999555555563, 0.6349206302847066, 0.5833333284722223, 0.35294117147058834, 0.7333333284222223, 0.959999995008, 0.23999999500800012, 0.639999995392, 0.7999999950888891, 0.6666666622222223, 0.5652173863137997, 0.4838709627471384, 0.6315789424930748, 0.34482758192627827, 0.6923076876701184, 0.5333333283555556, 0.22222221739369008, 0.43243242746530314, 0.7333333284222223, 0.7692307642603551, 0.5199999950080001, 0.5970149210692806, 0.39999999564800004, 0.2978723354277954, 0.7857142808163265, 0.27586206568370986, 0.576271181763861, 0.7407407359122085, 0.5660377308650766, 0.7999999950367347, 0.6666666620524693, 0.8648648599561725, 0.3749999953125, 0.4999999950000001, 0.3571428521428572, 0.3333333285802469, 0.6206896502259216, 0.4878048730517549, 0.3478260819659736, 0.31999999635200005, 0.8235294069204152, 0.7272727225309917, 0.42105262670360116, 0.4888888839111111, 0.4186046467712277, 0.2962962914677641, 0.4999999950000001, 0.5957446763603441, 0.2727272693847567, 0.7272727222830578, 0.5142857093877552, 0.25263157674016623, 0.2999999968, 0.7894736792243767, 0.4814814773113856, 0.29411764249134953, 0.4615384569656805, 0.34146341070791203, 0.5599999952000001, 0.7142857093112245, 0.8571428521541952, 0.6857142807183674, 0.7037036988751715, 0.5714285666805498, 0.7857142807142857, 0.4705882303114187, 0.37209301838831804, 0.3333333284222223, 0.8999999950500001, 0.4374999954882812, 0.0, 0.45833332855034725, 0.5614035037857804, 0.45945945507669844, 0.6363636317355372, 0.564102559184747, 0.7199999952000001, 0.31999999596800005, 0.33846153475029583, 0.16666666291666676, 0.16666666373299324, 0.3809523774258504, 0.7499999950347224, 0.35714285291454084, 0.782608690888469, 0.6585365808685306, 0.6428571378826532, 0.6382978673970123, 0.41176470228373707, 0.5490196030757402, 0.22580644661810625, 0.3783783736742148, 0.9285714235969389, 0.4878048735514575, 0.4615384565680473, 0.7692307642603551, 0.7407407357475996, 0.33333332888888895, 0.38596490824253615, 0.47999999512800007, 0.3870967691987514, 0.3380281652291212, 0.3636363602505166, 0.479999995072, 0.5581395306652245, 0.8571428521683674, 0.6249999950439454, 0.22033897925883375, 0.4615384565680473, 0.5490196029988467, 0.8571428521683674, 0.4109589008444362, 0.541666661701389, 0.3448275822116528, 0.5416666618836805, 0.13636363239669433, 0.8571428521428571, 0.5060240917694877, 0.9824561353524162, 0.9411764655882354, 0.7096774143600416, 0.42857142429705225, 0.4523809477891157, 0.5581395306652245, 0.35714285255102046, 0.6341463371326591, 0.45569619830155433, 0.8115941979836171, 0.6249999950195313, 0.3333333285802469, 0.7234042503214125, 0.3582089502784585, 0.2580645112591052, 0.3076923027218935, 0.44186046057328293, 0.5806451563371489, 0.6666666618666668, 0.5499999950500001, 0.8749999950781251, 0.6666666619135801, 0.9473684160526316, 0.8421052581717452, 0.4615384565680473, 0.5777777731950617, 0.5454545404958678, 0.49999999531250006, 0.999999995, 0.3529411716262976, 0.999999995, 0.599999995512, 0.823529406782007, 0.6153846104667982, 0.5384615335798818, 0.7096774143600416, 0.7878787829201102, 0.823529406782007, 0.7586206847086802, 0.699999995, 0.8648648598685172, 0.5217391262003781, 0.46153845669953986, 0.5714285668367348, 0.3999999955555556, 0.4666666622222223, 0.2909090867834711, 0.4285714244897959, 0.5853658486615111, 0.45161289864724247, 0.45714285217959183, 0.7906976694429422, 0.444444439808516, 0.5483870918210197, 0.40506328748597986, 0.6363636314049588, 0.9499999950125001, 0.9130434732703214, 0.6315789430470914, 0.7407407359122085, 0.999999995, 0.7692307642307693, 0.39999999502958583, 0.3636363594731405, 0.8823529361937716, 0.7142857093877552, 0.47619047174603185, 0.79999999505, 0.6315789423822715, 0.4999999950500001, 0.42307691855029594, 0.7499999950195314, 0.3333333290589569, 0.6666666617447917, 0.4615384570940172, 0.4285714239795918, 0.6440677916116059, 0.7894736792243767, 0.7368421006094183, 0.2592592560150892, 0.5569620206056722, 0.24489795465222827, 0.7333333283333333, 0.6666666622222223, 0.6666666618381345, 0.27272726921487606, 0.8333333283680556, 0.874999995, 0.5714285665306124, 0.2127659549117248, 0.9189189139225714, 0.5161290273881375, 0.6428571378635205, 0.6486486436815194, 0.588235289965398, 0.6666666618666668, 0.5714285666805498, 0.2962962913580247, 0.6896551674197384, 0.3823529372837371, 0.29787233593481216, 0.47999999564800006, 0.5333333285333334, 0.4255319098958805, 0.5263157845983379, 0.3243243209349891, 0.6666666617283951, 0.6249999950195313, 0.22727272363894632, 0.38596490824253615, 0.4444444395061729, 0.8333333283680556, 0.6874999951757813, 0.7692307642603551, 0.5405405355441928, 0.39999999520000007, 0.5999999958, 0.888888883888889, 0.742857138057143, 0.9387755052228238, 0.5925925882784637, 0.5333333294222222, 0.3636363603822314, 0.6999999950500001, 0.6666666616888889, 0.47826086551039704, 0.6956521691493385, 0.749999995, 0.6956521691493385, 0.7586206847086802, 0.7499999950781251, 0.3333333302864584, 0.5999999951125, 0.8461538412721893, 0.448275858234245, 0.8181818132231407, 0.36666666166666667, 0.7499999950781251, 0.26666666166666675, 0.36363635900826446, 0.5714285669556419, 0.5714285664540818, 0.5714285668367348, 0.7199999950079999, 0.27906976341806383, 0.44444444086666673, 0.7499999953125, 0.39999999501250005, 0.6222222177777779, 0.7037036987654323, 0.2325581365062196, 0.30508474232691757, 0.29411764233564014, 0.2962962914677641, 0.24657533767686257, 0.33333332958333334, 0.4705882303114187, 0.2790697644131963, 0.5555555513117285, 0.4864864819284149, 0.639999995648, 0.5937499950195314, 0.6086956472589792, 0.823529406782007, 0.4705882307266437, 0.6129032208116546, 0.9166666616666667, 0.49382715613473566, 0.2857142820861678, 0.5333333285333334, 0.28571428184807257, 0.452830183766465, 0.915254232301063, 0.5151515101928377, 0.5384615334615386, 0.9302325531638724, 0.4363636319603306, 0.5185185140740741, 0.559999995008, 0.7826086906616257, 0.5161290275130074, 0.6999999950500001, 0.7586206847086802, 0.4615384565680473, 0.7199999950079999, 0.68421052132964, 0.5599999952000001, 0.6769230724449705, 0.6111111061265434, 0.5454545408264463, 0.4545454497107439, 0.639999995072, 0.7272727223140496, 0.2962962913580247, 0.4545454497107439, 0.7027026977940103, 0.5454545404958678, 0.39999999520000007, 0.6190476141496599, 0.4999999959183673, 0.5384615339349113, 0.24999999730229594, 0.888888883888889, 0.3809523759637189, 0.7142857092857143, 0.7857142807397959, 0.47058823149557866, 0.22222221728395072, 0.28571428099773244, 0.38461537964497045, 0.7142857096938776, 0.43243242904309714, 0.5714285667120181, 0.4666666622222223, 0.35294117148788934, 0.6428571380165817, 0.6363636316219009, 0.4545454502066116, 0.879999995008, 0.3499999965125001, 0.44444444000000005, 0.3809523762358277, 0.7027026977063551, 0.41666666180555556, 0.3333333299153646, 0.4399999953920001, 0.34285713800408163, 0.8421052581578947, 0.4999999955165817, 0.653846148912722, 0.6666666617079889, 0.3529411728719723, 0.7999999950000002, 0.7142857093877552, 0.33333332836076823, 0.6153846108579881, 0.2647058796410035, 0.6896551674197384, 0.6363636320247934, 0.5263157848199447, 0.26086956253728216, 0.8823529361764706, 0.31999999596800005, 0.5238095195351474, 0.6666666617283951, 0.624999995, 0.47272726772892565, 0.999999995, 0.4799999952000001, 0.4999999950500001, 0.6666666619135801, 0.46153845669953986, 0.7058823479584776, 0.32653060749687635, 0.4545454502066116, 0.37209301858301785, 0.5714285664540818, 0.24242423797979806, 0.3720930183234181, 0.4230769197411243, 0.3157894687396123, 0.6785714235969388, 0.7450980342637449, 0.5230769185988166, 0.6086956472589792, 0.33870967460587936, 0.22857142448979595, 0.6451612853277836, 0.6046511585722013, 0.5555555508024692, 0.6666666621056242, 0.6499999952000001, 0.38709677026245815, 0.4761904725623583, 0.4827586164090369, 0.1913043453066163, 0.21874999658203131, 0.5454545405693297, 0.4186046466197945, 0.6086956471833649, 0.712328762281854, 0.46153845776114005, 0.34920634499370123, 0.5999999958, 0.6153846104142012, 0.5333333284222223, 0.7317073123140988, 0.8717948667981592, 0.6086956472589792, 0.4999999952000001, 0.5142857099755102, 0.4102564052859961, 0.44444444000000005, 0.8888888838957477, 0.6956521689224953, 0.4999999950347222, 0.6666666617555557, 0.49999999555555563, 0.6666666619501135, 0.7499999950195314, 0.5555555506944445, 0.6909090861884297, 0.5769230725221894, 0.9444444394444445, 0.6399999952000001, 0.7441860415359655, 0.79999999505, 0.9411764656055364, 0.5999999950500001, 0.5833333286458334, 0.4313725441138024, 0.5517241329369797, 0.4074074024691358, 0.8695652124007562, 0.874999995, 0.476190471473923, 0.7999999950617285, 0.6896551677051131, 0.874999995, 0.6976744140616551]
    def draw_hist(myList, Title, Xlabel, Ylabel, Xmin, Xmax, Ymin, Ymax):
        plt.hist(myList, 20)
        plt.xlabel(Xlabel)
        plt.xlim(Xmin, Xmax)
        plt.ylabel(Ylabel)
        plt.ylim(Ymin, Ymax)
        plt.title(Title)
        plt.show()
    draw_hist(rouge_score, 'ROUGEList', 'score', 'number', 0.0, 1, 0, 200)  # 直方图展示