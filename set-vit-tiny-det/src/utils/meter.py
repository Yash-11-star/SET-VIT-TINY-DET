"""
Metric Meter Utilities
Track training metrics and progress
"""

import math


class AverageMeter:
    """
    Track average value of a metric
    
    Example:
        loss_meter = AverageMeter()
        for batch in dataloader:
            loss = compute_loss(batch)
            loss_meter.update(loss.item(), n=batch_size)
        print(f"Average loss: {loss_meter.avg}")
    """
    
    def __init__(self, name='', fmt=':.4f'):
        """
        Initialize meter
        
        Args:
            name: Metric name
            fmt: Format string for printing
        """
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        """Reset meter"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        """
        Update meter with new value
        
        Args:
            val: New value
            n: Weight (usually batch size)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        """String representation"""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(
            name=self.name,
            val=self.val,
            avg=self.avg
        )


class ProgressMeter:
    """
    Track and display training progress
    
    Example:
        progress = ProgressMeter(num_batches, ['loss', 'top1', 'top5'])
        for batch_idx, (data, target) in enumerate(train_loader):
            loss = train_step(data, target)
            progress.update(batch_idx, loss.item())
            progress.display(batch_idx)
    """
    
    def __init__(self, num_batches, meters, prefix=""):
        """
        Initialize progress meter
        
        Args:
            num_batches: Total number of batches
            meters: List of AverageMeter objects
            prefix: Prefix for display
        """
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch):
        """
        Display progress
        
        Args:
            batch: Current batch number
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches):
        """Get format string for batch display"""
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'