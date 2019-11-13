package de.tum.in.probmodels.graph;

import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;
import de.tum.in.naturals.set.NatBitSet;
import de.tum.in.naturals.set.NatBitSets;
import it.unimi.dsi.fastutil.ints.Int2ObjectFunction;
import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntCollection;
import it.unimi.dsi.fastutil.ints.IntOpenHashSet;
import it.unimi.dsi.fastutil.ints.IntSet;
import it.unimi.dsi.fastutil.ints.IntStack;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.List;
import java.util.PrimitiveIterator;

/**
 * Finds the SCCs of a given graph / transition system using Tarjan's algorithm. Taken from owl.
 */
public final class SccDecomposition {
  // TODO Parallel tarjan?
  // TODO Optimize for only bsccs

  // Initial value for the low link - since we update the low-link whenever we find a link to a
  // state we can use this to detect trivial SCCs. MAX_VALUE is important for "<" comparisons
  static final int NO_LINK = Integer.MAX_VALUE;

  private final IntStack explorationStack = new IntArrayList();
  private final boolean includeTransient;
  private final Deque<TarjanState> path = new ArrayDeque<>();
  private final IntSet processedNodes = new IntOpenHashSet();
  private final List<NatBitSet> sccs = new ArrayList<>();
  private final Int2ObjectMap<TarjanState> stateMap = new Int2ObjectOpenHashMap<>();
  private final Int2ObjectFunction<? extends PrimitiveIterator.OfInt> successorFunction;
  private int index = 0;

  private SccDecomposition(Int2ObjectFunction<? extends PrimitiveIterator.OfInt> successorFunction,
      boolean includeTransient) {
    this.successorFunction = successorFunction;
    this.includeTransient = includeTransient;
  }

  public static List<NatBitSet> computeSccs(
      Int2ObjectFunction<? extends PrimitiveIterator.OfInt> function,
      IntCollection initialStates, boolean includeTransient) {
    SccDecomposition decomposition = new SccDecomposition(function, includeTransient);

    initialStates.forEach((int initialState) -> {
      if (!decomposition.stateMap.containsKey(initialState)
          && !decomposition.processedNodes.contains(initialState)) {
        decomposition.run(initialState);
      }
    });

    assert includeTransient
        || decomposition.sccs.stream().noneMatch(scc -> isTransient(function, scc));

    return Collections.unmodifiableList(decomposition.sccs);
  }


  public static boolean isTransient(Int2ObjectFunction<? extends PrimitiveIterator.OfInt> function,
      NatBitSet scc) {
    if (scc.size() > 1) {
      return false;
    }

    int state = Iterables.getOnlyElement(scc);
    return !Iterators.contains(function.apply(state), state);
  }

  public static boolean isBscc(Int2ObjectFunction<? extends PrimitiveIterator.OfInt> function,
      NatBitSet scc) {
    return scc.intStream().parallel().noneMatch(state -> {
      PrimitiveIterator.OfInt successors = function.apply(state);
      while (successors.hasNext()) {
        if (!scc.contains(successors.nextInt())) {
          return true;
        }
      }
      return false;
    });
  }

  private TarjanState create(int node) {
    assert !stateMap.containsKey(node) && !processedNodes.contains(node)
        : String.format("Node %s already processed", node);

    int nodeIndex = index;
    index += 1;

    PrimitiveIterator.OfInt successorIterator = successorFunction.apply(node);
    TarjanState state = new TarjanState(node, nodeIndex, successorIterator);

    explorationStack.push(node);
    stateMap.put(node, state);
    return state;
  }

  @SuppressWarnings("ObjectEquality")
  private void run(int initial) {
    assert path.isEmpty();
    TarjanState state = create(initial);

    //noinspection LabeledStatement - Without the label this method gets ugly
    outer:
    while (true) {
      int node = state.node;
      int nodeIndex = state.nodeIndex;

      PrimitiveIterator.OfInt successorIterator = state.successorIterator;
      while (successorIterator.hasNext()) {
        int successor = successorIterator.nextInt();

        if (node == successor) {
          if (state.lowLink == NO_LINK) {
            state.lowLink = nodeIndex;
          }
          // No need to process self-loops
          continue;
        }

        if (processedNodes.contains(successor)) {
          continue;
        }

        TarjanState successorState = stateMap.get(successor);
        assert successorState != state; // NOPMD

        if (successorState == null) {
          // Successor was not processed, do that now
          path.push(state);
          state = create(successor);
          //noinspection ContinueStatementWithLabel
          continue outer;
        }

        // Successor is not fully explored and we found a link to it, hence the low-link of this
        // state is less than or equal to the successors link.
        int successorIndex = successorState.nodeIndex;
        assert successorIndex != nodeIndex;

        int successorLowLink = successorState.getLowLink();
        if (successorLowLink < state.lowLink) {
          state.lowLink = successorLowLink;
        }

        assert state.lowLink <= nodeIndex;
      }

      // Finished handling this state by identifying whether it is a root of an SCC and
      // backtracking information if not. There are three possible cases:
      // 1) No link to this state has been found at all (-> transient SCC)
      // 2) This state is its own low-link (-> root of true SCC)
      // 3) State has true low link (-> non-root element of SCC)

      int lowLink = state.lowLink;
      if (lowLink == NO_LINK) {
        // This state has no back-link at all - transient state
        assert explorationStack.peekInt(0) == node;
        assert isTransient(successorFunction, NatBitSets.singleton(node));

        if (this.includeTransient) {
          sccs.add(NatBitSets.singleton(node));
        }

        explorationStack.popInt();
        processedNodes.add(node);
      } else if (lowLink == nodeIndex) {
        // This node can't reach anything younger than itself, thus by invariant it is the root of
        // an SCC. We now build the SCC and remove all now superfluous information (to keep the used
        // data-structures as small as possible)
        assert !explorationStack.isEmpty();

        // Gather all states in this SCC by popping the stack until we find the back-link
        NatBitSet scc;
        int stackNode = explorationStack.popInt();
        if (stackNode == node) { // NOPMD
          // Singleton SCC
          scc = NatBitSets.singleton(node);
          assert !isTransient(successorFunction, scc);
        } else {
          scc = NatBitSets.set();
          scc.add(stackNode);
          do {
            // Pop the stack until we find our node
            stackNode = explorationStack.popInt();
            scc.add(stackNode);
          } while (stackNode != node); // NOPMD
        }
        sccs.add(scc);

        // Remove all information about the popped states - retain the indices information since
        // we need to know which states have been processed.
        stateMap.keySet().removeAll(scc);
        processedNodes.addAll(scc);
      } else {
        // If this state is not a root, update the predecessor (which has to exist)
        assert !path.isEmpty() && lowLink < nodeIndex;

        TarjanState predecessorState = path.getFirst();
        // Since the current state has a "true" low-link, it is a possible low-link for the
        // predecessor, too. By invariant, it points to a non-finished state, i.e. a state in some
        // not yet found SCC.
        int predecessorLowLink = predecessorState.lowLink;

        if (lowLink < predecessorLowLink) {
          // Also happens if predecessor's low-link is NO_LINK - we may have found a back-edge to
          // the predecessor
          predecessorState.lowLink = lowLink;
        }
      }

      // Backtrack on the work-stack
      if (path.isEmpty()) {
        break;
      }
      state = path.pop();
    }

    //noinspection ConstantConditions
    assert path.isEmpty();
  }

  private static final class TarjanState {
    final int node;
    final int nodeIndex;
    final PrimitiveIterator.OfInt successorIterator;
    int lowLink;

    TarjanState(int node, int nodeIndex, PrimitiveIterator.OfInt successorIterator) {
      this.node = node;
      this.nodeIndex = nodeIndex;
      this.successorIterator = successorIterator;
      lowLink = NO_LINK;
    }

    int getLowLink() {
      // In standard Tarjan, all "NO_LINK"-states would have their own index as low-link.
      return lowLink == NO_LINK ? nodeIndex : lowLink;
    }

    @Override
    public String toString() {
      return nodeIndex + "(" + (lowLink == NO_LINK ? "X" : lowLink) + ") " + node;
    }
  }
}
