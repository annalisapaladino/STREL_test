class Reach(Node):
    """
    Reachability operator for STREL. Models bounded or unbounded reach
    over a spatial graph.
    """
    def __init__(
        self,
        phi_1: Node,
        phi_2: Node,
        distance_function,
        d1,
        d2,
        is_unbounded: bool = False,
        distance_domain_min=None,
        distance_domain_max=None,
        neighbors_fn=None
    ) -> None:
        super().__init__()
        self.phi_1 = phi_1
        self.phi_2 = phi_2
        self.distance_function = distance_function
        self.d1 = d1
        self.d2 = d2
        self.is_unbounded = is_unbounded
        self.distance_domain_min = distance_domain_min  # min(D)
        self.distance_domain_max = distance_domain_max  # max(D)
        self.neighbors_fn = neighbors_fn  # neighbors_fn(node) -> list of (neighbor, weight)

    def __str__(self) -> str:
        bound_type = "unbounded" if self.is_unbounded else f"[{self.d1},{self.d2}]"
        return f"({self.phi_1}) Reach{bound_type} ({self.phi_2})"

    def time_depth(self) -> int:
        return 0

    def _boolean(self, s1: dict, s2: dict, graph_nodes) -> dict:
        if self.is_unbounded:
            return self._unbounded_reach(s1, s2, graph_nodes)
        else:
            return self._bounded_reach(s1, s2, graph_nodes)

    def _bounded_reach(self, s1: dict, s2: dict, graph_nodes) -> dict:
        s = {} # Prova farlo un array
        Q = []

        for l in graph_nodes:
            if self.d1 == self.distance_domain_min:
                s[l] = s2[l]
            else:
                s[l] = self.distance_domain_min
            Q.append((l, s2[l], self.distance_domain_min))

        while Q:
            Q_prime = []
            for l, v, d in Q:
                for l_prime, w in self.neighbors_fn(l):
                    v_new = min(v, s1[l_prime])
                    d_new = d + self.distance_function(w)
                    if self.d1 <= d_new <= self.d2:
                        s[l_prime] = max(s[l_prime], v_new)
                    if d_new < self.d2:
                        found = False
                        for idx, (lp, vv, dd) in enumerate(Q_prime):
                            if lp == l_prime & dd==d_new:
                                Q_prime[idx] = (lp, max(vv, v_new), dd)
                                found = True
                                break
                        if not found:
                            Q_prime.append((l_prime, v_new, d_new))
            Q = Q_prime
        return s

    def _unbounded_reach(self, s1: dict, s2: dict, graph_nodes) -> dict:
        if self.d1 == self.distance_domain_min:
            return s2

        d_max = max(self.distance_function(w) for l in graph_nodes for _, w in self.neighbors_fn(l))
        d2_prime = self.d1 + d_max
        s = self._bounded_reach(s1, s2, graph_nodes)

        T = set(graph_nodes)
        while T:
            T_prime = set()
            for l in T:
                for l_prime, w in self.neighbors_fn(l):
                    v_new = s[l] * s1[l_prime] 
                    if v_new != s[l_prime]:
                        s[l_prime] = v_new
                        T_prime.add(l_prime)
            T = T_prime
        return s

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        raise NotImplementedError("Quantitative semantics for Reach not implemented in this version.")