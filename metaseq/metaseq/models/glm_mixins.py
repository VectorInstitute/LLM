import torch


def non_conflict(func):
    func.non_conflict = True
    return func


class BaseMixin(torch.nn.Module):
    non_conflict = non_conflict

    def __init__(self):
        super(BaseMixin, self).__init__()
        # define new params

    def reinit(self, parent_model=None):
        # reload the initial params from previous trained modules
        # you can also get access to other mixins through parent_model.get_mixin().
        pass

    # can define hook-functions here
    # ...

    # If the hook is just a pre- or post- transformation,
    # You can use @non_conflict to mark it,
    # and run `old_impl` to make it compatible with other mixins.
    # Eg.,
    #
    # @non_conflict
    # def attention_fn(q, k, v, mask, dropout_fn, old_impl=standard_attention, **kw_args):
    #     new_q, new_k, new_v = pre_hack(q, k, v)
    #     attn_result = old_impl(q, k, v, mask, dropout_fn, **kw_args)
    #     attn_result = post_hack(attn_result)
    #     return attn_result
