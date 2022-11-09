import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from src.tasks.supervised import ClassificationTask as _ClassificationTask
from torch.nn.utils.stateless import functional_call
from copy import deepcopy


class AlphaWeight(nn.Module):
    def __init__(self, base_weight, train_alpha=False, sample_alpha=True, init_alpha=0.,
                 sample_range=(0, 1), zero_weights=False):
        super().__init__()
        self.train_alpha = train_alpha
        self.sample_range = sample_range
        self.register_buffer('base_weight', base_weight.detach().clone())
        alpha = torch.full((1,), init_alpha, dtype=base_weight.dtype, device=base_weight.device)
        if train_alpha:
            self.alpha_weight = torch.nn.Parameter(alpha)
        else:
            self.register_buffer('alpha_weight', alpha)
        if zero_weights:
            base_weight.detach().zero_()
        self.sample_alpha = sample_alpha

    @property
    def alpha(self):
        if self.sample_alpha and self.training:
            self.alpha_weight.uniform_(*self.sample_range)
        return self.alpha_weight

    def forward(self, w):
        return (1. - self.alpha) * self.base_weight + self.alpha * w

    def inverse_right(self, w):
        if self.alpha == 0.:
            return torch.zeros_like(w)
        return (w - (1. - self.alpha) * self.base_weight) / self.alpha

    def set_alpha(self, alpha):
        with torch.no_grad():
            if torch.is_tensor(alpha):
                self.alpha_weight.copy_(alpha)
            else:
                self.alpha_weight.fill_(alpha)


def parameterize_model(model, **kwargs):
    for module in list(model.modules()):
        for name, param in list(module.named_parameters(recurse=False)):
            parametrize.register_parametrization(
                module, name, AlphaWeight(param, **kwargs))


def set_alpha_model(model, alpha, sample_alpha=None):
    for module_name, module in list(model.named_modules()):
        parametrizations = getattr(module, 'parametrizations', {})
        for param_name, param_modules in parametrizations.items():
            for pm in param_modules:
                if isinstance(pm, AlphaWeight):
                    if isinstance(alpha, dict):
                        alpha_value = alpha[f'{module_name}.{param_name}']
                    else:
                        alpha_value = alpha
                    pm.set_alpha(alpha_value)
                    if sample_alpha is not None:
                        pm.sample_alpha = sample_alpha


def get_alpha_model(model):
    alphas = {}
    for module_name, module in list(model.named_modules()):
        parametrizations = getattr(module, 'parametrizations', {})
        for param_name, param_modules in parametrizations.items():
            for pm in param_modules:
                if isinstance(pm, AlphaWeight):
                    alphas[f'{module_name}.{param_name}'] = pm.alpha_weight


def deparameterize_model(model, leave_parametrized=True):
    for module in list(model.modules()):
        parametrizations = getattr(module, 'parametrizations', {})
        for name in dict(parametrizations).keys():
            parametrize.remove_parametrizations(
                module, name, leave_parametrized=leave_parametrized)


def functional_mixed_model(inp, *models, weight=None, return_params_and_buffers=False):
    mixed_model = models[0]
    params_and_buffers = dict(mixed_model.named_buffers())
    models_params = [dict(m.named_parameters()) for m in models]
    for name, _ in mixed_model.named_parameters():
        params_and_buffers[name] = 0
        if isinstance(weight, dict):
            weight_vec = weight[name]
        else:
            weight_vec = weight
        for i in range(len(models_params)):
            params_and_buffers[name] += models_params[i][name] * weight_vec[i]
    out = functional_call(mixed_model, params_and_buffers, inp)
    if return_params_and_buffers:
        return out, params_and_buffers
    return out


class LearnedMixedModel(nn.Module):
    def __init__(self, models, train_base_models=False):
        super().__init__()
        self.models = nn.ModuleList(models)
        for model in self.models:
            for p in model.parameters():
                p.requires_grad_(train_base_models)
        self.parameter_names = []
        for name, _ in models[0].named_parameters():
            self.parameter_names.append(name)
        self.weight = nn.Parameter(torch.randn(len(self.parameter_names), len(models)))

    def forward(self, x):
        n_weights = {}
        weight = self.weight.softmax(dim=-1)
        for i, name in enumerate(self.parameter_names):
            n_weights[name] = weight[i]
        out, params_and_buffers = functional_mixed_model(
            x, *self.models, weight=n_weights, return_params_and_buffers=True)
        self.mixed_state_dict = params_and_buffers
        return out


class ClassificationTask(_ClassificationTask):
    def __init__(self, model, optimizer, use_alpha_weight=True, sample_range=(0., 1.), **kwargs):
        super().__init__(model, optimizer, **kwargs)
        self.use_alpha_weight = use_alpha_weight
        self.sample_range = sample_range

        parameterize_model(self.model, sample_range=self.sample_range)
        self.save_hyperparameters()

    def calibrate(self, loader, num_steps=100, num_models_keep=3):
        if self.global_step == 0:
            return
        set_alpha_model(self.model, alpha=1., sample_alpha=False)
        deparameterize_model(self.model, leave_parametrized=True)
        self.past_models = getattr(self, 'past_models', [])
        self.past_models.append(deepcopy(self.model).cpu())

        if len(self.past_models) > 1:
            # free past models
            del self.past_models[:-num_models_keep]
            model = LearnedMixedModel(self.past_models).to(self.device)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            print(f'Number of trainable parameters: {sum(p.numel() for p in trainable_params)}')
            optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
            model.train()
            for idx, batch in enumerate(loader):
                if idx > num_steps:
                    break
                x, yt = self.prepare_batch(batch)
                optimizer.zero_grad()
                with torch.enable_grad():
                    y = model(x)
                    loss = self.loss(y, yt)
                    loss.backward()
                optimizer.step()
            self.model.load_state_dict(model.mixed_state_dict)
            print(model.weight.softmax(dim=-1).mean(0))
            for i, m in enumerate(self.past_models):
                self.past_models[i] = m.to("cpu")
        out = super().calibrate(loader, num_steps)
        parameterize_model(self.model, init_alpha=0., sample_alpha=True,
                           sample_range=self.sample_range)
        return out
