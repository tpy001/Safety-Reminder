
from loguru import logger
from abc import ABC, abstractmethod

class UtilityEvaluator():
    def __init__(self,*args, **kwargs):
        pass

    @abstractmethod
    def judge(self, pred, data=None):
        pass

    def remove_special_chars(self, response):
        special_chars = ".,?!;:"
        res = []
        for i in range(len(response)):
            res.append( response[i].translate(str.maketrans('', '', special_chars)) )
        return res

    def evaluation(self,pred, data=None):
        pred = self.remove_special_chars(pred)
        res = self.judge(pred, data)
        correct = 0
        unknown = 0
        for i in range(len(res)):
            if res[i] == 'unknown':
                unknown += 1
            elif res[i] == 'yes':
                correct += 1

        ACC = round((correct / float(len(data))) * 100, 2)
        Unknown = round((unknown / float(len(data))) * 100, 2)
        print("ACC: ", ACC)
        print("Unknown: ", Unknown)
        return res
    
class MMBenchEvaluator(UtilityEvaluator):
    def judge(self, pred, data):
        answer = data['answer']
        assert len(answer) == len(pred)
        res = []
        for i in range(len(answer)):
            if len(pred[i]) == 1:
                if answer[i].upper() == pred[i][0].upper() : 
                    res.append('yes')
                else:
                    res.append('no')
            else:
                if answer[i].upper()  == pred[i][0].upper() : 
                    res.append('yes')
                else:
                    res.append('no')
        return res

    def evaluation(self,pred, data=None):
        pred = self.remove_special_chars(pred)
        res = self.judge(pred, data)
        
        calculator = AccuracyCalculator(res, data.data['category'])
        calculator.calculate_accuracy()
        calculator.print_results()
        return res

class MMEEvaluator(UtilityEvaluator):
    def judge(self, pred, data):
        res = []
        for i in range(len(data)):
            answer = data[i]['answer']
            response = pred[i]

            answer = answer.lower()
            response = response.lower()
            assert answer in ["yes", "no"] # gt can only be yes or no.

            flag = -1
            if response not in ["yes", "no"]:
                response = response[:4]
                if "yes" in response:
                    response = 'yes'
                elif "no" in response:
                    response = 'no'
            
            if response in ["yes", "no"]:
                if response == answer:
                    flag = 1
                else:
                    flag = 0
            else:
                flag = -1
                
            if flag == 1:  
                res.append('yes')
            elif flag == 0:
                res.append('no')
            else:
                res.append('unknown')
        return res

    def cal_acc(self, res, data):
        categories = set(data.data['category'])
        count_easy = {category: 0 for category in categories}
        count_hard = {category: 0 for category in categories}
        count_unknown = {category: 0 for category in categories}
        category_num = {category: 0 for category in categories}

        # 统计每个类别的正确预测、未知和总数
        for i in range(len(res)):
            category = data[i]['category']
            if category in categories:
                category_num[category] += 1
                if res[i] == 'unknown':
                    count_unknown[category] += 1
                elif res[i] == 'yes':
                    count_easy[category] += 1

        # 统计硬预测（连续两个都是 'yes'）
        for i in range(0, len(res) - 1, 2):
            category = data[i]['category']
            if category in categories:
                if res[i] == 'yes' and res[i + 1] == 'yes':
                    count_hard[category] += 2
                elif res[i] == 'unknown' or res[i + 1] == 'unknown':
                    count_unknown[category] += 2

        acc_easy = {}
        acc_hard = {}
        acc_unknown = {}
        acc_total = {
            'easy': 0,
            'hard': 0,
            'unknown': 0
        }

        total_samples = len(data)

        for category in categories:
            num = category_num[category]
            if num > 0:  # 避免除以零
                acc_easy[category] = round((count_easy[category] / float(num)) * 100, 2)
                acc_hard[category] = round((count_hard[category] / float(num)) * 100, 2)
                acc_unknown[category] = round((count_unknown[category] / float(num)) * 100, 2)

        # 总计准确率
        easy_total = sum(count_easy.values())
        hard_total = sum(count_hard.values())
        unknown_total = sum(count_unknown.values())

        acc_total['easy'] = round((easy_total / float(total_samples)) * 100, 2)
        acc_total['hard'] = round((hard_total / float(total_samples)) * 100, 2)
        acc_total['unknown'] = round((unknown_total / float(total_samples)) * 100, 2)

        return acc_easy, acc_hard, acc_unknown, acc_total

        
    def evaluation(self,pred, data=None):
        pred = self.remove_special_chars(pred)
        res = self.judge(pred, data)

        acc_easy, acc_hard, acc_unknown, acc_total = self.cal_acc(res,data)

        categories = sorted(set(data.data['category']))
        # print the results
        logger.info(f"{'Category':<20} {'Easy Acc':<20} {'Hard Acc':<20} {'Unknown Acc':<20}")
        logger.info("=" * 80)

        for category in categories:
            easy_acc = acc_easy[category]
            hard_acc = acc_hard[category]
            unknown_acc = acc_unknown[category]
            logger.info(f"{category:<20} {easy_acc:<20.2f} {hard_acc:<20.2f} {unknown_acc:<20.2f}")
        
        logger.info("=" * 80)
        logger.info(f"{'Total':<20} {acc_total['easy']:<20.2f} {acc_total['hard']:<20.2f} {acc_total['unknown']:<20.2f}")
        return res
    
