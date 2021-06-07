package de.tum.in.probmodels.model;

import com.google.common.collect.Sets;
import de.tum.in.naturals.set.NatBitSet;
import de.tum.in.naturals.set.NatBitSets;
import de.tum.in.naturals.unionfind.IntArrayUnionFind;
import de.tum.in.naturals.unionfind.IntUnionFind;
import explicit.SuccessorsIterator;
import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntAVLTreeSet;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntCollection;
import it.unimi.dsi.fastutil.ints.IntList;
import it.unimi.dsi.fastutil.ints.IntLists;
import it.unimi.dsi.fastutil.ints.IntOpenHashSet;
import it.unimi.dsi.fastutil.ints.IntSet;
import it.unimi.dsi.fastutil.ints.IntSortedSet;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.ListIterator;
import java.util.Objects;
import java.util.Set;
import java.util.function.IntUnaryOperator;
import java.util.function.Predicate;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import prism.ModelType;

public class CollapseView<M extends Model> extends AbstractModel implements CollapseModel<M> {
  private static final Logger logger = Logger.getLogger(CollapseView.class.getName());
  private final IntUnionFind collapseUF = new IntArrayUnionFind(0); // Maintains a union-find structure to store representatives of collapsed states.
  private final NatBitSet removedStates = NatBitSets.set();
  private final NatBitSet representativeStates = NatBitSets.set();
  private final M model;
  private final Int2ObjectMap<List<Distribution>> overwrite = new Int2ObjectOpenHashMap<>(); // Implements a cache to avoid repeated computation of successors
  private final IntSet overwriteCacheValid = new IntOpenHashSet(); // Membership in this indicates if the overwrite cache holds a valid distribution value (It could have been changed recently)

  // Returns underlying model
  public CollapseView(M model) {
    this.model = model;
  }

  @Override
  public ModelType getModelType() {
    return model.getModelType();
  }

  @Override
  // Returns number of states in current view of the model, excludes removed states
  public int getNumStates() {
    return model.getNumStates() - removedStates.size();
  }

  @Override
  public int getNumTransitions() {
    throw new UnsupportedOperationException();
  }

  @Override
  public IntCollection getInitialStates() {
    IntCollection initialStates = model.getInitialStates();
    IntSortedSet states = new IntAVLTreeSet();
    initialStates.forEach((int state) -> states.add(representative(state)));
    return states;
  }

  @Override
  public boolean isInitialState(int state) {
    return model.isInitialState(representative(state));
  }

  // Computed a new distribution for the given state. The map function helps in mapping the distribution for removed states to representative states
  // the unchanged predicate returns false for a distribution if it might have an undesired state. If this is true, distribution is not recomputed
  // Eg. If a distribution for a state might have a removed state, the distribution needs to computed.
  private List<Distribution> computeSuccessors(int state, IntUnaryOperator map,
      Predicate<Distribution> unchanged) {
    List<Distribution> distributions = overwrite.get(state);
    if (distributions == null) {
      distributions = new ArrayList<>(model.getChoices(state));
    }

    // the list of distributions is returned only if there are any changes. anyDifferent keeps track of this.
    boolean anyDifferent = false;
    ListIterator<Distribution> iterator = distributions.listIterator();
    while (iterator.hasNext()) {
      Distribution distribution = iterator.next();

      if (distribution.isEmpty()) {
        anyDifferent = true;
        iterator.remove();
      }
      // Don't do anything if distribution is unchanged.
      else if (unchanged.test(distribution)) {
        assert Objects.equals(distribution, distribution.map(map).scaled());
      }
      // The support of the distribution must have changed, and it must be rebuilt
      else {
        // Returns a new distribution builder object according to the given map and the distribution
        DistributionBuilder builder = distribution.map(map);
        if (builder.isEmpty()) {
          anyDifferent = true;
          iterator.remove();
        } else {
          // A scaled distribution is built, i.e., if the sum of the probabilities is less than 1, the distribution is scaled
          Distribution scaled = builder.scaled();
          anyDifferent = anyDifferent || !scaled.equals(distribution);
          // Replaces "distribution" with "scaled"
          iterator.set(scaled);
        }
      }
    }

    // Ensures that all distributions obey the unchanged criteria
    assert distributions.stream().allMatch(unchanged);
    // Ensures that all distributions obey the mapping function
    assert distributions.stream().allMatch(d -> d.equals(d.map(map).build()));
    // Ensures that no distribution is empty
    assert distributions.stream().noneMatch(Distribution::isEmpty);
    // Ensures that no distribution's support consists of the state itself.
    assert distributions.stream().noneMatch(d -> d.contains(state));
    return anyDifferent ? distributions : null;
  }

  @Override
  public List<Distribution> getChoices(int state) {
    assert !isRemoved(state);

    List<Distribution> distributions = null;
    // This is true only when state is not already in overwriteCacheValid. It means that the overwrite map contains the latest distributions and recomputation is not required.
    // Further, it now marks that the state's entry in overwrite is now valid. (overwrite is updated in the block)
    if (overwriteCacheValid.add(state)) {
      // Creates a map from a state to it's representative. Avoids self loops
      IntUnaryOperator map = successor -> {
        int representative = representative(successor);
        // Checking if the state hasn't been removed
        assert !isRemoved(representative);
        return representative == state ? -1 : representative;
      };

      // Marks a distribution as changed if the support consists of a removed state or the state itself.
      Predicate<Distribution> unchanged = d ->
          !d.containsOneOf(removedStates) && !d.contains(state);
      // Calculates the distributions
      distributions = computeSuccessors(state, map, unchanged);
      if (distributions != null) {
        // The below 3 lines remove any duplicate distributions
        Set<Distribution> uniqueDistributions = new HashSet<>(distributions.size());
        Predicate<Distribution> filter = uniqueDistributions::add;
        distributions.removeIf(filter.negate());
        // The overwrite cache of state is now updated
        overwrite.put(state, distributions);
      }
    }

    // Gets the distribution in case the cache is valid
    if (distributions == null) {
      distributions = overwrite.get(state);
      if (distributions == null) {
        distributions = model.getChoices(state);
      }
    }

    assert distributions != null;
    // Ensures that no distribution is empty
    assert distributions.stream().noneMatch(Distribution::isEmpty);
    // Ensures that no distribution's support consists of the state itself.
    assert distributions.stream().noneMatch(d -> d.contains(state)) : state + " " + distributions;
    // Ensures that no distribution's support consists of a removed state
    assert distributions.stream().noneMatch(d -> d.containsOneOf(removedStates));
    // Ensures that all distribution's support consists of states that are their own representatives. Basically re-ensures that there are no removed states
    assert distributions.stream().map(Distribution::support).flatMap(Collection::stream)
        .allMatch(s -> s == representative(s));

    return Collections.unmodifiableList(distributions);
  }

  @Override
  public Distribution getChoice(int state, int action) {
    return getChoices(state).get(action);
  }

  @Override
  // getChoices returns a list of distributions. getActions returns a list of actions by converting the output of getChoices
  public List<Action> getActions(int state) {
    return getChoices(state).stream().map(Action::of).collect(Collectors.toList());
  }

  @Override
  public SuccessorsIterator getSuccessors(int s) {
    List<Distribution> choices = getChoices(s);
    if (choices.isEmpty()) {
      return SuccessorsIterator.empty();
    }
    if (choices.size() == 1) {
      return SuccessorsIterator.from(choices.get(0).support().iterator(), true);
    }

    NatBitSet union = NatBitSets.set();
    choices.forEach(d -> union.or(d.support()));
    return SuccessorsIterator.from(union.iterator(), true);
  }


  @Override
  // Given a partitioning of states, collapses each partition. Returns a list of representatives
  public IntList collapse(List<? extends IntSet> stateList) {
    if (stateList.isEmpty()) {
      //noinspection AssignmentOrReturnOfFieldWithMutableType
      return IntLists.EMPTY_LIST;
    }

    // Ensures that no 2 partitions intersect
    assert stateList.stream().allMatch(states ->
        stateList.stream().noneMatch(others -> !others.equals(states)
            && !Sets.intersection(states, others).isEmpty()));

    // Collects all states of stateList
    NatBitSet newCollapsed = NatBitSets.set();
    stateList.forEach(newCollapsed::or);

    // Only collapse states of the collapsed mode. Ensures state list doesn't have any removed states
    assert !newCollapsed.intersects(removedStates);

    // Collapse the states. representatives consists of the newly computed representatives
    IntList representatives = new IntArrayList(stateList.size());
    for (IntSet states : stateList) {
      // collapse individual partition 'states'
      representatives.add(collapse(states));
    }
    // Ensures that the class variable has all representatives. Must be added by private collapse function
    assert representativeStates.containsAll(representatives);

    // Representatives are consistent
    // Ensures that the computed representative of a state lies in the same set in stateList
    assert stateList.stream().allMatch(states ->
        states.stream().mapToInt(this::representative).allMatch(states::contains));
    // Ensures that each partition in stateList has only one representative
    assert stateList.stream().allMatch(states ->
        states.stream().mapToInt(this::representative).distinct().count() == 1L);
    // Ensures that no 2 partitions have the same representative.
    assert stateList.stream().flatMap(Collection::stream).mapToInt(this::representative)
        .distinct().count() == stateList.size();

    // Process the distributions for each partition
    for (int i = 0; i < stateList.size(); i++) {
      IntSet states = stateList.get(i);
      int representative = representatives.getInt(i);
      assert states.contains(representative)
          && representative == representative(representative);

      // Set of newly calculated distributions
      Collection<Distribution> collapsedDistributions = new HashSet<>();

      // Creates a map from a state to it's representative. Avoids self loops
      IntUnaryOperator map = successor -> {
        int successorRepresentative = this.representative(successor);
        // Checking if the state hasn't been removed
        assert !removedStates.contains(successorRepresentative);
        return successorRepresentative == representative ? -1 : successorRepresentative;
      };

      // Recalculating distributions
      states.forEach((int state) -> {
        // Marks a distribution as changed if the support consists of a removed state or the state itself.
        Predicate<Distribution> unchanged = d ->
            !d.containsOneOf(removedStates) && !d.contains(representative);
        // Recalculates the distributions
        var distributions = computeSuccessors(state, map, unchanged);
        if (distributions == null) {
          distributions = overwrite.get(state);
          if (distributions == null) {
            distributions = model.getChoices(state);
          }
        }
        collapsedDistributions.addAll(distributions);
        // Removes all states from overwrite cache. (Representative states are added later)
        overwrite.remove(state);
      });

      // No internal transitions
      // No distribution's support consists of a removed state
      assert collapsedDistributions.stream().noneMatch(d -> d.containsOneOf(removedStates));
      // No distribution has a self loop
      assert collapsedDistributions.stream().noneMatch(d -> d.contains(representative));
      // All states are cleared
      assert states.stream().mapToInt(Integer::intValue)
          .mapToObj(overwrite::get).allMatch(Objects::isNull);

      // updating overwrite cache
      overwrite.put(representative, new ArrayList<>(collapsedDistributions));
    }

    // overwrite doesn't contain a removed state
    assert overwrite.keySet().stream().noneMatch(this::isRemoved);

    // Remap all transitions. Other states might be pointing to some now merged state - we have
    // to update them too.
    overwriteCacheValid.clear();
    overwriteCacheValid.addAll(representatives);

    if (logger.isLoggable(Level.INFO)) {
      int transitionCount = 0;
      int actionCount = 0;
      int maxTransitions = 0;
      int maxActions = 0;
      int stateCount = representatives.size();
      for (int representative : representatives) {
        List<Distribution> distributions = getChoices(representative);
        if (distributions == null) {
          continue;
        }
        actionCount += distributions.size();
        maxActions = Math.max(maxActions, distributions.size());
        for (Distribution distribution : distributions) {
          transitionCount += distribution.size();
          maxTransitions = Math.max(maxTransitions, distribution.size());
        }
      }
      logger.info(String.format("Collapsed states: %d, Actions: %.2f avg/%d max, "
              + "Transitions %.2f avg/%d max", stateCount, actionCount / (double) stateCount,
          maxActions, transitionCount / (double) stateCount, maxTransitions));
    }

    return representatives;
  }

  // collapses a single set of states. Returns a representative
  private int collapse(IntSet states) {
    assert !states.isEmpty();
    if (states.size() == 1) {
      int state = states.iterator().nextInt();
      representativeStates.set(state);
      return state;
    }
    int modelStates = model.getNumStates();
    // Expanding size of collapseUF array. (There may be newly explored states in the model)
    if (collapseUF.size() <= modelStates) {
      collapseUF.add(modelStates * 2 + 1);
    }

    int anyState = representative(states.iterator().nextInt());
    // Updates collapseUF array. Does union operation for all states
    states.forEach((int state) -> collapseUF.union(anyState, state));
    // Gets a random state as a representative
    int representative = representative(anyState);

    // Remove all input states
    removedStates.or(states);
    // Mark representative state as not removed
    removedStates.clear(representative);
    representativeStates.set(representative);

    return representative;
  }

  @Override
  // If the state represents an MEC, then return the MEC number to which it belongs, otherwise, the state is its own representative
  public int representative(int state) {
    if (state >= collapseUF.size()) {  // the state doesn't have an entry in collapseUF, it must not have been collapsed until now and would be it's own representative
      return state;
    }
    // Returns the parent of the state in the Union-find tree.
    return collapseUF.find(state);
  }

  @Override
  public boolean isRemoved(int state) {
    assert (representative(state) != state) == removedStates.contains(state);
    return representative(state) != state;
  }

  @Override
  public NatBitSet removedStates() {
    return removedStates;
  }

  @Override
  public M getModel() {
    return model;
  }


  // Mutators

  @Override
  public void setInitialStates(Collection<Integer> initialStates) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void addChoice(int state, Distribution distribution) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void addChoice(int state, Action action) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void setChoice(int state, int action, Distribution distribution) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void setActions(int state, List<Action> actions) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void addInitialState(int i) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void clearState(int i) {
    throw new UnsupportedOperationException();
  }

  @Override
  public int addState() {
    throw new UnsupportedOperationException();
  }

  @Override
  public void addStates(int numToAdd) {
    throw new UnsupportedOperationException();
  }
}
