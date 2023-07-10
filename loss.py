from torch.nn import CrossEntropyLoss
from torch.nn.modules import loss
from utils.TripletLoss import TripletLoss,MGN_TripletLoss,CrossEntropyLabelSmooth
from opt import opt

class Loss(loss._Loss):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, outputs, labels):
        cross_entropy_loss = CrossEntropyLabelSmooth(num_classes=opt.num_classes)
        #cross_entropy_loss = CrossEntropyLoss()
        triplet_loss = MGN_TripletLoss(margin=opt.margin)

        Triplet_Loss = [triplet_loss(output, labels) for output in outputs[1:2]]
        Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)

        CrossEntropy_Loss = [cross_entropy_loss(output, labels) for output in outputs[2:]]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        loss_sum = (opt.triplet_lambda * Triplet_Loss) +  (opt.softmax_lambda * CrossEntropy_Loss)

        print('\rCurrent Step:  CrossEntropy_Loss:%.4f   Triplet_Loss:%.4f  total loss:%.4f    ' % (
            CrossEntropy_Loss.data.cpu().numpy(),
            Triplet_Loss.data.cpu().numpy(),
              loss_sum.data.cpu().numpy()),end='')
        return loss_sum, Triplet_Loss, CrossEntropy_Loss
