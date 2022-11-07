# SAKD
Our code is divided by dataset, and we upload our original records.


Baseline performance of all ANN models 
 
 Model  | CIFAR10 | CIFAR100  | ImageNet  
  ---- | ----- | ------ | ------ 
 ResNet-18  | 95.65 | 78.35 | 71.5 
 ResNet-34  | 95.69 | 79.30 | 76.4
 ResNet-50  | 95.85 | 80.49 | 80.4
 ResNet-101  | 95.92 | 80.83 | 81.5
 ResNet-152  | 96.38 | 81.36 | 82.0
 ResNet-19  | 96.30 | 80.48 | ---

Baseline performance of all SNN models (200epoch for CIFAR , 120epoch for ImageNet)
 
 Model  | CIFAR10 | CIFAR100  | ImageNet  
  ---- | ----- | ------ | ------ 
 ResNet-18  | 94.33 | 75.03 | 61.13
 ResNet-34  | 93.50 | 71.72 | 64.40
 ResNet-50  | 93.54 | 70.34 | 67.45
 ResNet-101  | 55.83 | 35.72 | ---
 ResNet-152  | 10.00 | 10.00 | ---
 ResNet-19  | 94.28 | 78.28 | ---
 Sew18 | --- |--- |61.55

How to set up the environment ?
Our code is based on a specific version of spikingjelly
```
pip install spikingjelly==0.0.0.0.12
pip install cupy-cuda113
```
We use triangle surrogate gradients without cupy backend in the framework, the environment can be set in the following way:
Copy the code at the end into the framework's surrogate.py


some possible problems

If you encounter difficulties in this process, you can directly replace
```
neuron.MultiStepLIFNode(surrogate_function=surrogate.Tri(alpha=1), backend='cupy', decay_input=False)
```
in the model.py with
```
neuron.MultiStepLIFNode(backend='cupy', decay_input=False)
```
to use the default surrogate gradient.

If you have difficulty installing cupy, you can use the default backend
```
neuron.MultiStepLIFNode(decay_input=False)
```

---------
```
class tri(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * (ctx.alpha - input.abs()).clamp(min=0)
        return grad_x, None

class Tri(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)


    @staticmethod
    def spiking_function(x, alpha):
        return tri.apply(x, alpha)
    '''
    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return (x * alpha).sigmoid()
    '''
    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        alpha = str(self.alpha) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''
        if dtype == 'fp32':
            code += f'''
            {tab4_str}const float {sg_name}_x_abs = 1.0f - fabsf({x});
            float {y};
            if ({sg_name}_x_abs > 0.0f)
            {curly_bracket_l}
                {y} = {sg_name}_x_abs;
            {curly_bracket_r}
            else
            {curly_bracket_l}
                {y} = 0.0f;
            {curly_bracket_r}
            '''

         elif dtype == 'fp16':
            code += f'''
            {tab4_str}const half2 {sg_name}_x_abs = __hsub2(__float2half2_rn(1.0f), __habs2({x}));
            {tab4_str}const half2 {sg_name}_x_abs_ge_w = __hge2({sg_name}_x_abs, __float2half2_rn(0.0f));
            {tab4_str}const half2 {y} = __hadd2(__hmul2({sg_name}_x_abs,  {sg_name}_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), {sg_name}_x_abs_ge_w), __float2half2_rn(0.0f)));
            '''
        else:
            raise NotImplementedError
        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code
```
