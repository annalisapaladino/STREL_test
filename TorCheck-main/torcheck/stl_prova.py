 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
# Temporary Version with distance_matrix... more updates to be done
# ==============================================================================

# ==============================================================================
# Copyright 2020-* Luca Bortolussi. All Rights Reserved.
# Copyright 2020-* Laura Nenzi.     All Rights Reserved.
# Copyright 2020-* AI-CPS Group @ University of Trieste. All Rights Reserved.
# ==============================================================================

"""A fully-differentiable implementation of Signal Temporal Logic semantic trees."""

from typing import Union

# For custom type-hints
# For tensor functions
import torch
import torch.nn.functional as F
from torch import Tensor

# Custom types
realnum = Union[float, int] # Defines a custom type alias for any real number (used for thresholds)


# TODO: automatic check of timespan when evaluating robustness? (should be done only at root node)

def eventually(x: Tensor, time_span: int) -> Tensor:
    # TODO: as of this implementation, the time_span must be int (we are working with steps,
    #  not exactly points in the time axis)
    # TODO: maybe converter from resolution to steps, if one has different setting
    """
    STL operator 'eventually' in 1D.

    Parameters
    ----------
    x: torch.Tensor
        Signal
    time_span: any numeric type
        Timespan duration

    Returns
    -------
    torch.Tensor
    A tensor containing the result of the operation.
    """
    return F.max_pool1d(x, kernel_size=time_span, stride=1)
    # Defines an eventually operator using 1D max pooling:
    #    Applies over time.
    #    For each window of time_span, takes the maximum value.
    #    Used in quantitative semantics to implement F and G

class Node:
    """Abstract node class for STL semantics tree."""

    def __init__(self) -> None:
        # Must be overloaded.
        pass

    def __str__(self) -> str:
        # Must be overloaded.
        pass

    def boolean(self, x: Tensor, evaluate_at_all_times: bool = False) -> Tensor:
        """
        Evaluates the boolean semantics at the node.

        Parameters
        ----------
        x : torch.Tensor, of size N_samples x N_vars x N_sampling_points
            The input signals, stored as a batch tensor with trhee dimensions.
        evaluate_at_all_times: bool
            Whether to evaluate the semantics at all times (True) or
            just at t=0 (False).

        Returns
        -------
        torch.Tensor
        A tensor with the boolean semantics for the node.
        """
        z: Tensor = self._boolean(x)
        if evaluate_at_all_times:
            return z
        else:
            return self._extract_semantics_at_time_zero(z)

        # Calls internal _boolean method to evaluate Boolean semantics.
        # By default returns value at time t=0 unless evaluate_at_all_times=True.

    def quantitative(
            self,
            x: Tensor,
            normalize: bool = False,
            evaluate_at_all_times: bool = False,
    ) -> Tensor:
        """
        Evaluates the quantitative semantics at the node.

        Parameters
        ----------
        x : torch.Tensor, of size N_samples x N_vars x N_sampling_points
            The input signals, stored as a batch tensor with three dimensions.
        normalize: bool
            Whether the measure of robustness if normalized (True) or
            not (False). Currently not in use.
        evaluate_at_all_times: bool
            Whether to evaluate the semantics at all times (True) or
            just at t=0 (False).

        Returns
        -------
        torch.Tensor
        A tensor with the quantitative semantics for the node.
        """
        z: Tensor = self._quantitative(x, normalize) # normalize is a placeholder
        if evaluate_at_all_times:
            return z
        else:
            return self._extract_semantics_at_time_zero(z)
        
        # Same idea as boolean(...) but for robustness evaluation.
        
    def set_normalizing_flag(self, value: bool = True) -> None:
        """
        Setter for the 'normalization of robustness of the formula' flag.
        Currently not in use.
        """

    def time_depth(self) -> int:
        """Returns time depth of bounded temporal operators only."""
        # Must be overloaded.

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        """Private method equivalent to public one for inner call."""
        # Must be overloaded.

    def _boolean(self, x: Tensor) -> Tensor:
        """Private method equivalent to public one for inner call."""
        # Must be overloaded.

    @staticmethod
    def _extract_semantics_at_time_zero(x: Tensor) -> Tensor:
        """Extrapolates the vector of truth values at time zero"""
        return torch.reshape(x[:, 0, 0], (-1,))


class Atom(Node):
    """Atomic formula node; for now of the form X<=t or X>=t"""

    def __init__(self, var_index: int, threshold: realnum, lte: bool = False) -> None:
        super().__init__()
        self.var_index: int = var_index
        self.threshold: realnum = threshold
        self.lte: bool = lte

    def __str__(self) -> str:
        s: str = (
                "x_"
                + str(self.var_index)
                + (" <= " if self.lte else " >= ")
                + str(round(self.threshold, 4))
        )
        return s

    def time_depth(self) -> int:
        return 0

    def _boolean(self, x: Tensor) -> Tensor: # Boolean: checks if x[var_index] <= threshold
        # extract tensor of the same dimension as data, but with only one variable
        xj: Tensor = x[:, self.var_index, :]
        xj: Tensor = xj.view(xj.size()[0], 1, -1)
        if self.lte:
            z: Tensor = torch.le(xj, self.threshold)
        else:
            z: Tensor = torch.ge(xj, self.threshold)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor: # Quantitative: returns the signed distance from the threshold (e.g. how much above or below)
        # extract tensor of the same dimension as data, but with only one variable
        xj: Tensor = x[:, self.var_index, :]
        xj: Tensor = xj.view(xj.size()[0], 1, -1)
        if self.lte:
            z: Tensor = -xj + self.threshold
        else:
            z: Tensor = xj - self.threshold
        if normalize:
            z: Tensor = torch.tanh(z)
        return z


class Not(Node):
    """Negation node."""

    def __init__(self, child: Node) -> None:
        super().__init__()
        self.child: Node = child

    def __str__(self) -> str:
        s: str = "not ( " + self.child.__str__() + " )"
        return s

    def time_depth(self) -> int:
        return self.child.time_depth()

    def _boolean(self, x: Tensor) -> Tensor:
        z: Tensor = ~self.child._boolean(x)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z: Tensor = -self.child._quantitative(x, normalize) # negates the robustness score
        return z


class And(Node):
    """Conjunction node."""

    def __init__(self, left_child: Node, right_child: Node) -> None:
        super().__init__()
        self.left_child: Node = left_child
        self.right_child: Node = right_child

    def __str__(self) -> str:
        s: str = (
                "( "
                + self.left_child.__str__()
                + " and "
                + self.right_child.__str__()
                + " )"
        )
        return s

    def time_depth(self) -> int:
        return max(self.left_child.time_depth(), self.right_child.time_depth())

    def _boolean(self, x: Tensor) -> Tensor:
        z1: Tensor = self.left_child._boolean(x)
        z2: Tensor = self.right_child._boolean(x)
        size: int = min(z1.size()[2], z2.size()[2])
        z1: Tensor = z1[:, :, :size]
        z2: Tensor = z2[:, :, :size]
        z: Tensor = torch.logical_and(z1, z2)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1: Tensor = self.left_child._quantitative(x, normalize)
        z2: Tensor = self.right_child._quantitative(x, normalize)
        size: int = min(z1.size()[2], z2.size()[2])
        z1: Tensor = z1[:, :, :size]
        z2: Tensor = z2[:, :, :size]
        z: Tensor = torch.min(z1, z2)
        return z


class Or(Node):
    """Disjunction node."""

    def __init__(self, left_child: Node, right_child: Node) -> None:
        super().__init__()
        self.left_child: Node = left_child
        self.right_child: Node = right_child

    def __str__(self) -> str:
        s: str = (
                "( "
                + self.left_child.__str__()
                + " or "
                + self.right_child.__str__()
                + " )"
        )
        return s

    def time_depth(self) -> int:
        return max(self.left_child.time_depth(), self.right_child.time_depth())

    def _boolean(self, x: Tensor) -> Tensor:
        z1: Tensor = self.left_child._boolean(x)
        z2: Tensor = self.right_child._boolean(x)
        size: int = min(z1.size()[2], z2.size()[2])
        z1: Tensor = z1[:, :, :size]
        z2: Tensor = z2[:, :, :size]
        z: Tensor = torch.logical_or(z1, z2)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1: Tensor = self.left_child._quantitative(x, normalize)
        z2: Tensor = self.right_child._quantitative(x, normalize)
        size: int = min(z1.size()[2], z2.size()[2])
        z1: Tensor = z1[:, :, :size]
        z2: Tensor = z2[:, :, :size]
        z: Tensor = torch.max(z1, z2)
        return z


class Globally(Node): # ALWAYS operator: the formula is always true in the time span
    """Globally node."""

    def __init__(
            self,
            child: Node,
            unbound: bool = False,
            right_unbound: bool = False,
            left_time_bound: int = 0,
            right_time_bound: int = 1,
            adapt_unbound: bool = True,
    ) -> None:
        super().__init__()
        self.child: Node = child
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1
        self.adapt_unbound: bool = adapt_unbound

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = "always" + s0 + " ( " + self.child.__str__() + " )"
        return s

    def time_depth(self) -> int:
        if self.unbound:
            return self.child.time_depth()
        elif self.right_unbound:
            return self.child.time_depth() + self.left_time_bound
        else:
            # diff = torch.le(torch.tensor([self.left_time_bound]), 0).float()
            return self.child.time_depth() + self.right_time_bound - 1
            # (self.right_time_bound - self.left_time_bound + 1) - diff

    def _boolean(self, x: Tensor) -> Tensor:
        z1: Tensor = self.child._boolean(x[:, :, self.left_time_bound:])  # nested temporal parameters
        # z1 = z1[:, :, self.left_time_bound:]
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummin(torch.flip(z1, [2]), dim=2)
                z: Tensor = torch.flip(z, [2])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.min(z1, 2, keepdim=True)
        else:
            z: Tensor = torch.ge(1.0 - eventually((~z1).double(), self.right_time_bound - self.left_time_bound), 0.5)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1: Tensor = self.child._quantitative(x[:, :, self.left_time_bound:], normalize)
        # z1 = z1[:, :, self.left_time_bound:]
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummin(torch.flip(z1, [2]), dim=2)
                z: Tensor = torch.flip(z, [2])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.min(z1, 2, keepdim=True)
        else:
            z: Tensor = -eventually(-z1, self.right_time_bound - self.left_time_bound)
        return z


class Eventually(Node): # EVENTUALLY operator: the formula is eventually true (true at any time) in the time span
    """Eventually node."""

    def __init__(
            self,
            child: Node,
            unbound: bool = False,
            right_unbound: bool = False,
            left_time_bound: int = 0,
            right_time_bound: int = 1,
            adapt_unbound: bool = True,
    ) -> None:
        super().__init__()
        self.child: Node = child
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1
        self.adapt_unbound: bool = adapt_unbound

        if (self.unbound is False) and (self.right_unbound is False) and \
                (self.right_time_bound <= self.left_time_bound):
            raise ValueError("Temporal thresholds are incorrect: right parameter is higher than left parameter")

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = "eventually" + s0 + " ( " + self.child.__str__() + " )"
        return s

    # TODO: coherence between computation of time depth and time span given when computing eventually 1d
    def time_depth(self) -> int:
        if self.unbound:
            return self.child.time_depth()
        elif self.right_unbound:
            return self.child.time_depth() + self.left_time_bound
        else:
            # diff = torch.le(torch.tensor([self.left_time_bound]), 0).float()
            return self.child.time_depth() + self.right_time_bound - 1
            # (self.right_time_bound - self.left_time_bound + 1) - diff

    def _boolean(self, x: Tensor) -> Tensor:
        z1: Tensor = self.child._boolean(x[:, :, self.left_time_bound:])
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummax(torch.flip(z1, [2]), dim=2)
                z: Tensor = torch.flip(z, [2])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.max(z1, 2, keepdim=True)
        else:
            z: Tensor = torch.ge(eventually(z1.double(), self.right_time_bound - self.left_time_bound), 0.5)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1: Tensor = self.child._quantitative(x[:, :, self.left_time_bound:], normalize)
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummax(torch.flip(z1, [2]), dim=2)
                z: Tensor = torch.flip(z, [2])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.max(z1, 2, keepdim=True)
        else:
            z: Tensor = eventually(z1, self.right_time_bound - self.left_time_bound)
        return z


class Until(Node):
    # UNTIL operator: phi_1 U[a, b] phi_2: 
    #   phi_2 must be true sometime in [a, b]
    #   phi_1 must be true at all time before that
    
    # TODO: maybe define timed and untimed until, and use this class to wrap them
    # TODO: maybe faster implementation (of untimed until especially)
    """Until node."""

    def __init__(
            self,
            left_child: Node, # phi_1
            right_child: Node, # phi_2
            unbound: bool = False, # phi_1 U phi_2 over infinite time
            right_unbound: bool = False, # only upper bound is infinite
            left_time_bound: int = 0, # define the interval [a, b]
            right_time_bound: int = 1, # define the interval [a, b]
    ) -> None:
        super().__init__()
        self.left_child: Node = left_child
        self.right_child: Node = right_child
        self.unbound: bool = unbound # Unbounded: full future
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1

        if (self.unbound is False) and (self.right_unbound is False) and \
                (self.right_time_bound <= self.left_time_bound):
            raise ValueError("Temporal thresholds are incorrect: right parameter is higher than left parameter")

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = "( " + self.left_child.__str__() + " until" + s0 + " " + self.right_child.__str__() + " )"
        return s

    def time_depth(self) -> int: # Calculates how far in time this operator needs to look to compute its output
        sum_children_depth: int = self.left_child.time_depth() + self.right_child.time_depth()
        if self.unbound:
            return sum_children_depth
        elif self.right_unbound:
            return sum_children_depth + self.left_time_bound
        else:
            # diff = torch.le(torch.tensor([self.left_time_bound]), 0).float()
            return sum_children_depth + self.right_time_bound - 1
            # (self.right_time_bound - self.left_time_bound + 1) - diff

    def _boolean(self, x: Tensor) -> Tensor:
        if self.unbound:
            # this is phi_1 U phi_2 over the full future: 
            # We compute, for each time step, whether phi_2 becomes true in the future, and phi_1 holds up until then 
            z1: Tensor = self.left_child._boolean(x) # Get Boolean values for both operands
            z2: Tensor = self.right_child._boolean(x)
            # Shape: [batch, 1, time]
            size: int = min(z1.size()[2], z2.size()[2]) 
            z1: Tensor = z1[:, :, :size]
            z2: Tensor = z2[:, :, :size]
            
            # Builds a "history matrix" for each time step: was phi_1 true from now until t?
            z1_rep = torch.repeat_interleave(z1.unsqueeze(2), z1.unsqueeze(2).shape[-1], 2)
            z1_tril = torch.tril(z1_rep.transpose(2, 3), diagonal=-1)
            z1_triu = torch.triu(z1_rep)
            z1_def = torch.cummin(z1_tril + z1_triu, dim=3)[0]
            # same for z2
            z2_rep = torch.repeat_interleave(z2.unsqueeze(2), z2.unsqueeze(2).shape[-1], 2)
            z2_tril = torch.tril(z2_rep.transpose(2, 3), diagonal=-1)
            z2_triu = torch.triu(z2_rep)
            z2_def = z2_tril + z2_triu
            
            # Compute min(phi_1 segment, phi_2) across time window. Then take max over possible satisfaction points
            z: Tensor = torch.max(torch.min(torch.cat([z1_def.unsqueeze(-1), z2_def.unsqueeze(-1)], dim=-1), dim=-1)[0],
                                  dim=-1)[0] # This gives the truth value of the Until for each time point.
        elif self.right_unbound:
            # φ1 U[a,b] φ2 ≡ G[0,a](φ1) ∧ F[a,b](φ2) ∧ F[a,b](φ1 U φ2)
            timed_until: Node = And(Globally(self.left_child, left_time_bound=0, right_time_bound=self.left_time_bound),
                                    And(Eventually(self.right_child, right_unbound=True,
                                                   left_time_bound=self.left_time_bound),
                                        Eventually(Until(self.left_child, self.right_child, unbound=True),
                                                   left_time_bound=self.left_time_bound, right_unbound=True)))
            z: Tensor = timed_until._boolean(x)
        else:
            timed_until: Node = And(Globally(self.left_child, left_time_bound=0, right_time_bound=self.left_time_bound),
                                    And(Eventually(self.right_child, left_time_bound=self.left_time_bound,
                                                   right_time_bound=self.right_time_bound - 1),
                                        Eventually(Until(self.left_child, self.right_child, unbound=True),
                                                   left_time_bound=self.left_time_bound, right_unbound=True)))
            z: Tensor = timed_until._boolean(x)
            # It rewrites bounded until in terms of:
            #   Globally: ensures φ1 holds at start
            #   Eventually(φ2): ensures φ2 happens
            #   Eventually(UnboundedUntil): handles general satisfaction
        
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        if self.unbound:
            z1: Tensor = self.left_child._quantitative(x, normalize)
            z2: Tensor = self.right_child._quantitative(x, normalize)
            size: int = min(z1.size()[2], z2.size()[2])
            z1: Tensor = z1[:, :, :size]
            z2: Tensor = z2[:, :, :size]

            z1_rep = torch.repeat_interleave(z1.unsqueeze(2), z1.unsqueeze(2).shape[-1], 2)
            z1_tril = torch.tril(z1_rep.transpose(2, 3), diagonal=-1)
            z1_triu = torch.triu(z1_rep)
            z1_def = torch.cummin(z1_tril + z1_triu, dim=3)[0]

            z2_rep = torch.repeat_interleave(z2.unsqueeze(2), z2.unsqueeze(2).shape[-1], 2)
            z2_tril = torch.tril(z2_rep.transpose(2, 3), diagonal=-1)
            z2_triu = torch.triu(z2_rep)
            z2_def = z2_tril + z2_triu
            z: Tensor = torch.max(torch.min(torch.cat([z1_def.unsqueeze(-1), z2_def.unsqueeze(-1)], dim=-1), dim=-1)[0],
                                  dim=-1)[0]
            # z: Tensor = torch.cat([torch.max(torch.min(
            #    torch.cat([torch.cummin(z1[:, :, t:].unsqueeze(-1), dim=2)[0], z2[:, :, t:].unsqueeze(-1)], dim=-1),
            #    dim=-1)[0], dim=2, keepdim=True)[0] for t in range(size)], dim=2)
        elif self.right_unbound:
            timed_until: Node = And(Globally(self.left_child, left_time_bound=0, right_time_bound=self.left_time_bound),
                                    And(Eventually(self.right_child, right_unbound=True,
                                                   left_time_bound=self.left_time_bound),
                                        Eventually(Until(self.left_child, self.right_child, unbound=True),
                                                   left_time_bound=self.left_time_bound, right_unbound=True)))
            z: Tensor = timed_until._quantitative(x, normalize=normalize)
        else:
            timed_until: Node = And(Globally(self.left_child, left_time_bound=0, right_time_bound=self.left_time_bound),
                                    And(Eventually(self.right_child, left_time_bound=self.left_time_bound,
                                                   right_time_bound=self.right_time_bound - 1),
                                        Eventually(Until(self.left_child, self.right_child, unbound=True),
                                                   left_time_bound=self.left_time_bound, right_unbound=True)))
            z: Tensor = timed_until._quantitative(x, normalize=normalize)
        return z

class Since(Node):
    """Since node."""

    # SINCE operator: phi_1 U[a, b] phi_2: 
    #   phi_2 held within [a, b]
    #   phi_1 held from then until now

    # STL doesn’t natively support past, but we can simulate it by flipping time and reusing Until
    
    def __init__(
            self,
            left_child: Node,
            right_child: Node,
            unbound: bool = False,
            right_unbound: bool = False,
            left_time_bound: int = 0,
            right_time_bound: int = 1,
    ) -> None:
        super().__init__()
        self.left_child: Node = left_child
        self.right_child: Node = right_child
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1

        if (not self.unbound) and (not self.right_unbound) and \
                (self.right_time_bound <= self.left_time_bound):
            raise ValueError("Temporal thresholds are incorrect: right parameter is higher than left parameter")

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = f"( {self.left_child} since{s0} {self.right_child} )"
        return s

    def time_depth(self) -> int:
        sum_children_depth: int = self.left_child.time_depth() + self.right_child.time_depth()
        if self.unbound:
            return sum_children_depth
        elif self.right_unbound:
            return sum_children_depth + self.left_time_bound
        else:
            return sum_children_depth + self.right_time_bound - 1

    def _boolean(self, x: Tensor) -> Tensor:
        # Past-time: need to flip the input
        x_flipped = torch.flip(x, [2]) # reverse along time axis

        # Reuse Until semantics on flipped signal
        until_node = Until( # construct a matching Until
            self.left_child,
            self.right_child,
            unbound=self.unbound,
            right_unbound=self.right_unbound,
            left_time_bound=self.left_time_bound,
            right_time_bound=self.right_time_bound - 1,
        )

        # Compute on flipped signal
        z_flipped = until_node._boolean(x_flipped)

        # Flip back
        return torch.flip(z_flipped, [2]) # flip back

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        # Past-time: need to flip the input
        x_flipped = torch.flip(x, [2])

        until_node = Until(
            self.left_child,
            self.right_child,
            unbound=self.unbound,
            right_unbound=self.right_unbound,
            left_time_bound=self.left_time_bound,
            right_time_bound=self.right_time_bound - 1,
        )

        z_flipped = until_node._quantitative(x_flipped, normalize=normalize)

        return torch.flip(z_flipped, [2])

class Reach(Node):
    """
    Reachability operator for STREL. Models bounded or unbounded reach
    over a spatial graph.
    """
    def __init__(
        self,
        left_child: Node,
        right_child: Node,
        distance_matrix: Tensor,
        d1: realnum,
        d2: realnum,
        graph_nodes: Tensor,
        is_unbounded: bool = False,
        distance_domain_min: realnum = None,
        distance_domain_max: realnum = None,
    ) -> None:
        super().__init__()
        self.left_child = left_child
        self.right_child = right_child
        self.d1 = d1
        self.d2 = d2
        self.is_unbounded = is_unbounded
        self.distance_domain_min = distance_domain_min
        self.distance_domain_max = distance_domain_max
        self.graph_nodes = graph_nodes

        self.weight_matrix = distance_matrix
        self.adjacency_matrix = (distance_matrix > 0).int()

        self.boolean_min_satisfaction = torch.tensor(0.0)
        self.quantitative_min_satisfaction = torch.tensor(float('-inf'))

    def __str__(self) -> str:
        bound_type = "unbounded" if self.is_unbounded else f"[{self.d1},{self.d2}]"
        return f"Reach{bound_type}"

    def neighbors_fn(self, node):
        mask = self.adjacency_matrix[:, node] > 0
        neighbors = self.graph_nodes[mask]
        neigh_pairs = [(i.item(), self.weight_matrix[i, node].item()) for i in neighbors]
        # print('node = ', node, ' has neigh_pairs = ', neigh_pairs)
        return neigh_pairs

    def distance_function(self, weight):
        return weight

    # ----------------
    #    Boolean
    # ----------------

    def _boolean(self, x: Tensor) -> Tensor:
        s1 = self.left_child._boolean(x).squeeze()
        s2 = self.right_child._boolean(x).squeeze()

        if self.is_unbounded:
            return self._unbounded_reach_boolean(s1, s2)
        else:
            return self._bounded_reach_boolean(s1, s2)

    def _bounded_reach_boolean(self, s1, s2):
        s = torch.zeros(len(self.graph_nodes), dtype=torch.float32, requires_grad=True)

        for i, lt in enumerate(self.graph_nodes):
            l = lt.item()
            if self.d1 == self.distance_domain_min:
                s = s.clone().scatter_(0, torch.tensor([l]), s2[l].to(dtype=s.dtype))
            else:
                s = s.clone().scatter_(0, torch.tensor([l]), self.boolean_min_satisfaction.to(dtype=s.dtype))

        Q = {llt.item(): [(s2[llt.item()].to(dtype=s.dtype), self.distance_domain_min)] for llt in self.graph_nodes}

        while Q:
            Q_prime = {}
            for l in Q.keys():
                for v, d in Q[l]:
                    for l_prime, w in self.neighbors_fn(l):
                        v_new = torch.minimum(v, s1[l_prime].to(dtype=s.dtype))
                        d_new = d + w

                        if self.d1 <= d_new <= self.d2:
                            current_val = s[l_prime]
                            new_val = torch.maximum(current_val, v_new)
                            s = s.clone().scatter_(0, torch.tensor([l_prime]), new_val.to(dtype=s.dtype))

                        if d_new < self.d2:

                            existing_entries = Q_prime.get(l_prime, [])
                            updated = False
                            new_entries = []
                            for vv, dd in existing_entries:
                                if dd == d_new:
                                    new_v = torch.maximum(vv, v_new)
                                    new_entries.append((new_v, dd))
                                    updated = True
                                else:
                                    new_entries.append((vv, dd))

                            if not updated:
                                new_entries.append((v_new, d_new))
                            Q_prime[l_prime] = new_entries

            Q = Q_prime
        return s.unsqueeze(0).unsqueeze(-1)

    def _unbounded_reach_boolean(self, s1, s2):

        if self.d1 == self.distance_domain_min:
            s = s2.to(dtype=torch.float32)
        else:
            d_max = torch.max(self.distance_function(self.weight_matrix))
            self.d2 = self.d1 + d_max
            s = self._bounded_reach_boolean(s1, s2).squeeze()

        T = [n for n in self.graph_nodes]

        while T:
            T_prime = []

            for l in T:
                for l_prime, w in self.neighbors_fn(l):
                    v_prime = torch.minimum(s[l], s1[l_prime].to(dtype=s.dtype))
                    v_prime = torch.maximum(v_prime, s[l_prime])

                    if v_prime != s[l_prime]:
                        s = s.clone().scatter_(0, torch.tensor([l_prime]), v_prime.to(dtype=s.dtype))
                        T_prime.append(l_prime)

            T = T_prime

        return s.unsqueeze(0).unsqueeze(-1)

    # ---------------------
    #     Quantitative
    # ---------------------

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        s1 = self.left_child._quantitative(x, normalize).squeeze()
        s2 = self.right_child._quantitative(x, normalize).squeeze()

        if self.is_unbounded:
            return self._unbounded_reach_quantitative(s1, s2)
        else:
            return self._bounded_reach_quantitative(s1, s2)

    def _bounded_reach_quantitative(self, s1, s2):
        s = torch.full((len(self.graph_nodes),), self.quantitative_min_satisfaction, dtype=torch.float32, requires_grad=True)

        for i, lt in enumerate(self.graph_nodes):
            l = lt.item()
            if self.d1 == self.distance_domain_min:
                s = s.clone().scatter_(0, torch.tensor([l]), s2[l].to(dtype=s.dtype))
            else:
                s = s.clone().scatter_(0, torch.tensor([l]), self.quantitative_min_satisfaction.to(dtype=s.dtype))

        Q = {llt.item(): [(s2[llt.item()].to(dtype=s.dtype), self.distance_domain_min)] for llt in self.graph_nodes}

        while Q:
            Q_prime = {}
            for l in Q.keys():
                for (v, d) in Q[l]:
                    for l_prime, w in self.neighbors_fn(l):
                        v_new = torch.minimum(v, s1[l_prime].to(dtype=s.dtype))
                        d_new = d + w

                        if self.d1 <= d_new <= self.d2:
                            current_val = s[l_prime]
                            new_val = torch.maximum(current_val, v_new)
                            s = s.clone().scatter_(0, torch.tensor([l_prime]), new_val.to(dtype=s.dtype))

                        if d_new < self.d2:
                            existing_entries = Q_prime.get(l_prime, [])
                            updated = False
                            new_entries = []
                            for (vv, dd) in existing_entries:
                                if dd == d_new:
                                    new_v = torch.maximum(vv, v_new)
                                    new_entries.append((new_v, dd))
                                    updated = True
                                else:
                                    new_entries.append((vv, dd))
                            if not updated:
                                new_entries.append((v_new, d_new))
                            Q_prime[l_prime] = new_entries
            Q = Q_prime

        return s.unsqueeze(0).unsqueeze(-1)

    def _unbounded_reach_quantitative(self, s1, s2):

        if self.d1 == self.distance_domain_min:
            s = s2.to(dtype=torch.float32)
        else:
            d_max = torch.max(self.distance_function(self.weight_matrix))
            self.d2 = self.d1 + d_max
            s = self._bounded_reach_quantitative(s1, s2).squeeze()

        T = [n for n in self.graph_nodes]

        while T:
            T_prime = []

            for l in T:
                for l_prime, w in self.neighbors_fn(l):
                    v_prime = torch.minimum(s[l], s1[l_prime].to(dtype=s.dtype))
                    new_val = torch.maximum(v_prime, s[l_prime])
                    if new_val != s[l_prime]:
                        s = s.clone().scatter_(0, torch.tensor([l_prime]), new_val.to(dtype=s.dtype))
                        T_prime.append(l_prime)
            T = T_prime

        return s.unsqueeze(0).unsqueeze(-1)

class Escape(Node):
    """
    Escape operator for STREL. Models escape condition over a spatial graph.
    """
    def __init__(
        self,
        child: Node,
        distance_matrix: Tensor,
        d1: realnum,
        d2: realnum,
        graph_nodes: Tensor,
        distance_domain_min: realnum = None,
        distance_domain_max: realnum = None,
    ) -> None:
        super().__init__()
        self.child = child
        self.d1 = d1
        self.d2 = d2
        self.distance_domain_min = distance_domain_min
        self.distance_domain_max = distance_domain_max
        self.graph_nodes = graph_nodes

        self.weight_matrix = distance_matrix
        self.adjacency_matrix = (distance_matrix > 0).int()

        self.boolean_min_satisfaction = torch.tensor(0.0)
        self.quantitative_min_satisfaction = torch.tensor(float('-inf'))

    def neighbors_fn(self, node):
        mask = (self.adjacency_matrix[:, node] > 0)
        neighbors = self.graph_nodes[mask]
        return [(i.item(), self.weight_matrix[i, node].item()) for i in neighbors]

    def forward_neighbors_fn(self, node):
        mask = (self.adjacency_matrix[node, :] > 0)
        neighbors = self.graph_nodes[mask]
        return [(j.item(), self.weight_matrix[node, j].item()) for j in neighbors]

    def compute_min_distance_matrix(self):

        n = len(self.graph_nodes) # Numero di nodi nel grafo
        D = torch.full((n, n), float('inf')) # matrice n x n inizializzata con infinito (inf). Alla fine conterrà le distanze minime tra ogni coppia di nodi.

        for start in range(n): # Calcola le distanze dal nodo start a tutti gli altri
            visited = torch.zeros(n, dtype=torch.bool) # tiene traccia dei nodi già visitati
            distance = torch.full((n,), float('inf')) # array delle distanze minime dal nodo start
            distance[start] = 0 # Inizia da 'start'

            frontier = torch.zeros(n, dtype=torch.bool) # Nodi da visitare nel livello corrente
            frontier[start] = True # Inizia da 'start'

            while frontier.any(): # itera finché ci sono nodi nella frontiera
                next_frontier = torch.zeros(n, dtype=torch.bool) # Prepara la frontier per il livello successivo

                for node in torch.nonzero(frontier).flatten(): # Per ogni nodo nella frontiera attuale, lo segna come visitato
                    node = node.item()
                    visited[node] = True

                    for neighbor, weight in self.forward_neighbors_fn(node): # Recupera i vicini del nodo e il peso del collegamento
                        if visited[neighbor]:
                            continue
                        # edge_cost = weight if getattr(self, "use_weights", False) else 1.0 # Se use_weights=True, usa il peso reale dell’arco. Altrimenti, ogni arco conta 1 (hop count booleano)
                        edge_cost = weight # if self.use_weights else 1.0
                        new_dist = distance[node] + edge_cost # Calcola la distanza cumulativa provvisoria al vicino

                        # Se è una distanza più breve di quella attuale, la aggiorna e aggiunge il vicino alla prossima frontiera
                        if new_dist < distance[neighbor]:
                            distance[neighbor] = new_dist
                            next_frontier[neighbor] = True

                frontier = next_frontier # Passa alla prossima frontiera da esplorare

            D[start] = distance # Dopo aver esplorato tutti i percorsi dal nodo start, salva le distanze finali

        return D # Restituisce la matrice completa D con le distanze minime tra tutti i nodi, pesate o in hop

    # ----------------
    #    Boolean
    # ----------------

    def _boolean(self, x: Tensor) -> Tensor:

        s1 = self.child._boolean(x).squeeze()

        L = self.graph_nodes
        n = len(L)

        D = self.compute_min_distance_matrix()

        e = torch.ones((n, n), requires_grad=True) * self.boolean_min_satisfaction
        e = e - torch.diag(torch.diag(e)) + torch.diag(s1)

        T = [(i, i) for i in range(n)]

        while T:
            T_prime = []
            e_prime = e.clone()

            for l1, l2 in T:
                for l1_prime, w in self.neighbors_fn(l1):
                    new_val = torch.minimum(s1[l1_prime], e[l1, l2])
                    old_val = e[l1_prime, l2]
                    combined = torch.maximum(old_val, new_val)

                    if combined != old_val:
                        e_prime = e_prime.clone().index_put_(tuple(torch.tensor([[l1_prime], [l2]])), combined)
                        T_prime.append((l1_prime, l2))

            T = T_prime
            e = e_prime

        s = torch.ones(n, requires_grad=True) * self.boolean_min_satisfaction
        for i in range(n):
            vals = [e[i, j] for j in range(n) if self.d1 <= D[i, j] <= self.d2]
            if vals:
                max_val = torch.stack(vals).max()
                s = s.clone().scatter_(0, torch.tensor([i]), max_val.unsqueeze(0))

        return s.unsqueeze(0).unsqueeze(-1)

    # ---------------------
    #     Quantitative
    # ---------------------

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
      
        s1 = self.child._quantitative(x, normalize).squeeze()

        L = self.graph_nodes
        n = len(L)

        D = self.compute_min_distance_matrix()

        e = torch.ones((n, n), requires_grad=True) * self.quantitative_min_satisfaction
        e = e - torch.diag(torch.diag(e)) + torch.diag(s1)

        T = [(i, i) for i in range(n)]

        while T:
            T_prime = []
            e_prime = e.clone()

            for l1, l2 in T:
                for l1_prime, w in self.neighbors_fn(l1):
                    new_val = torch.minimum(s1[l1_prime], e[l1, l2])
                    old_val = e[l1_prime, l2]
                    combined = torch.maximum(old_val, new_val)

                    if combined != old_val:
                        e_prime = e_prime.clone().index_put_(
                            tuple(torch.tensor([[l1_prime], [l2]])), combined
                        )
                        T_prime.append((l1_prime, l2))

            T = T_prime
            e = e_prime

        s = torch.ones(n, requires_grad=True) * self.quantitative_min_satisfaction
        for i in range(n):
            vals = [
                e[i, j] for j in range(n)
                if self.d1 <= D[i, j] <= self.d2
            ]
            if vals:
                max_val = torch.stack(vals).max()
                s = s.clone().scatter_(0, torch.tensor([i]), max_val.unsqueeze(0))

        return s.unsqueeze(0).unsqueeze(-1)

class Somewhere(Node):
    """
    Somewhere operator for STREL. Models existence of a satisfying location within a distance interval.
    """
    def __init__(
        self,
        child: Node,
        distance_matrix: Tensor,
        d2: realnum,
        graph_nodes: Tensor,
        distance_domain_min: realnum = None,
        distance_domain_max: realnum = None,
    ) -> None:
        super().__init__()
        self.child = child
        self.d1 = 0
        self.d2 = d2
        self.distance_domain_min = distance_domain_min
        self.distance_domain_max = distance_domain_max
        self.graph_nodes = graph_nodes

        # Create a true node (always true)
        self.true_node = Atom(0, float('inf'), lte=True)  # x_0 <= inf (always true)
        
        # Create Reach operator
        self.reach_op = Reach(
            left_child=self.true_node,
            right_child=child,
            distance_matrix=distance_matrix,
            d1=self.d1,
            d2=d2,
            graph_nodes=graph_nodes,
            distance_domain_min=distance_domain_min,
            distance_domain_max=distance_domain_max
        )

    def __str__(self) -> str:
        return f"somewhere_[{self.d1},{self.d2}] ( {self.child} )"

    def _boolean(self, x: Tensor) -> Tensor:
        return self.reach_op._boolean(x)

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        return self.reach_op._quantitative(x, normalize)

class Everywhere(Node):
    """
    Everywhere operator for STREL. Models satisfaction of a property at all locations within a distance interval.
    """
    def __init__(
        self,
        child: Node,
        distance_matrix: Tensor,
        d2: realnum,
        graph_nodes: Tensor,
        distance_domain_min: realnum = None,
        distance_domain_max: realnum = None,
    ) -> None:
        super().__init__()
        self.child = child
        self.d1 = 0
        self.d2 = d2
        self.distance_domain_min = distance_domain_min
        self.distance_domain_max = distance_domain_max
        self.graph_nodes = graph_nodes

        # Create a true node (always true)
        self.true_node = Atom(0, float('inf'), lte=True)  # x_0 <= inf (always true)
        
        # Create Reach operator
        self.reach_op = Reach(
            left_child=self.true_node,
            right_child=Not(self.child), # child,
            distance_matrix=distance_matrix,
            d1=self.d1,
            d2=d2,
            graph_nodes=graph_nodes,
            distance_domain_min=distance_domain_min,
            distance_domain_max=distance_domain_max
        )

    def __str__(self) -> str:
        return f"somewhere_[{self.d1},{self.d2}] ( {self.child} )"

    def _boolean(self, x: Tensor) -> Tensor:
        return 1.0 - self.reach_op._boolean(x)

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        return - self.reach_op._quantitative(x, normalize)
    
class Surround(Node):
    """
    Surround operator for STREL. Models being surrounded by φ2 while in φ1 with distance constraints.
    """
    def __init__(
        self,
        left_child: Node,
        right_child: Node,
        distance_matrix: Tensor,
        d2: realnum,
        graph_nodes: Tensor,
        distance_domain_min: realnum = None,
        distance_domain_max: realnum = None,
    ) -> None:
        super().__init__()
        self.left_child = left_child
        self.right_child = right_child
        self.d1 = 0
        self.d2 = d2
        self.distance_domain_min = distance_domain_min
        self.distance_domain_max = distance_domain_max
        self.graph_nodes = graph_nodes
        
        # Create Reach operator
        self.reach_op = Reach(
            left_child=left_child,
            right_child=Not(Or(left_child, right_child)), 
            distance_matrix=distance_matrix,
            d1=self.d1,
            d2=d2,
            graph_nodes=graph_nodes,
            distance_domain_min=distance_domain_min,
            distance_domain_max=distance_domain_max
        )
        
        # Create Escape operator
        self.escape_op = Escape(
            child=left_child, 
            distance_matrix=distance_matrix,
            d1=d2,
            d2=distance_domain_max,
            graph_nodes=graph_nodes,
            distance_domain_min=distance_domain_min,
            distance_domain_max=distance_domain_max
        )

    def __str__(self) -> str:
        return f"somewhere_[{self.d1},{self.d2}] ( {self.child} )"

    def _boolean(self, x: Tensor) -> Tensor:
        
        s1 = self.left_child._boolean(x).squeeze()
        
        reach_part = 1.0 - self.reach_op._boolean(x).squeeze()
        
        escape_part = 1.0 - self.escape_op._boolean(x).squeeze()
        
        result = torch.minimum(s1, torch.minimum(reach_part, escape_part))
        return result.unsqueeze(0).unsqueeze(-1)

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        
        s1 = self.left_child._quantitative(x).squeeze()
        
        reach_part = - self.reach_op._quantitative(x).squeeze()
        
        escape_part = - self.escape_op._quantitative(x).squeeze()
        
        result = torch.minimum(s1, torch.minimum(reach_part, escape_part))
        return result.unsqueeze(0).unsqueeze(-1)
