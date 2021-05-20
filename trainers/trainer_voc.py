from trainers.base_trainer import BaseTrainer
from evaluator.voceval_crps import EvaluatorVOC_CRPS
from evaluator.voceval import EvaluatorVOC
class Trainer(BaseTrainer):
  def __init__(self, args, model, optimizer,lrscheduler):
    super().__init__(args, model, optimizer,lrscheduler)
    self.evaluation_method = "crps"

  def _get_loggers(self):
    super()._get_loggers()
    self.evaluation_method = "crps"
    #print(self.evaluation_method)
    if self.evaluation_method == "crps":
        self.TESTevaluator = EvaluatorVOC_CRPS(anchors=None,
                                          cateNames=self.labels,
                                          rootpath=self.dataset_root,
                                          score_thres=0.01,
                                          iou_thres=self.args.EVAL.iou_thres,
                                          use_07_metric=False
                                          )
        self.logger_custom = ['CRPS']+['CRPS@{}'.format(cls) for cls in self.labels]
    else:
        self.TESTevaluator = EvaluatorVOC(anchors=None,
                                          cateNames=self.labels,
                                          rootpath=self.dataset_root,
                                          score_thres=0.01,
                                          iou_thres=self.args.EVAL.iou_thres,
                                          use_07_metric=False
                                          )
        self.logger_custom = ['mAP']+['AP@{}'.format(cls) for cls in self.labels]
