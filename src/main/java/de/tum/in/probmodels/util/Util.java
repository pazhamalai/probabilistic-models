package de.tum.in.probmodels.util;

import de.tum.in.naturals.set.NatBitSet;
import de.tum.in.naturals.set.NatBitSets;
import de.tum.in.naturals.unionfind.IntUnionFind;
import de.tum.in.probmodels.model.Action;
import de.tum.in.probmodels.model.Model;
import it.unimi.dsi.fastutil.ints.Int2ObjectAVLTreeMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import it.unimi.dsi.fastutil.ints.IntIterator;
import java.awt.Color;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.function.IntFunction;
import java.util.function.IntPredicate;
import javax.annotation.Nullable;

public final class Util {
  public static final double DEFAULT_PRECISION = 1e-6;
  private static final double MACHINE_EPS = 1e-15;

  private Util() {
    // Empty
  }

  public static List<NatBitSet> unionFindPartition(IntIterator iterator, IntUnionFind unionFind) {
    Int2ObjectMap<NatBitSet> unionRootMap = new Int2ObjectAVLTreeMap<>();
    List<NatBitSet> foundBitSets = new ArrayList<>();
    while (iterator.hasNext()) {
      int state = iterator.nextInt();
      unionRootMap.computeIfAbsent(unionFind.find(state), s -> {
        NatBitSet result = NatBitSets.set();
        foundBitSets.add(result);
        return result;
      }).set(state);
    }
    return foundBitSets;
  }

  public static void modelToDotFile(String filename, Model model, IntFunction<Object> label,
      @Nullable IntFunction<Color> stateColors, IntPredicate stateFilter,
      @Nullable IntPredicate highlight) {
    StringBuilder dotString = new StringBuilder("digraph Model {\n\tnode [shape=box];\n");

    for (int state = 0; state < model.getNumStates(); state++) {
      if (!stateFilter.test(state)) {
        continue;
      }

      dotString.append(state).append(" [style=filled fillcolor=\"");
      if (highlight != null && highlight.test(state)) {
        dotString.append("#22CC22");
      } else if (model.isInitialState(state)) {
        dotString.append("#9999CC");
      } else if (stateColors != null) {
        Color color = stateColors.apply(state);
        dotString.append(String.format("#%02X%02X%02X",
            color.getRed(), color.getGreen(), color.getBlue()));
      } else {
        dotString.append("#DDDDDD");
      }

      dotString.append("\",label=\"").append(state);
      Object stateLabel = label.apply(state);
      if (stateLabel != null) {
        dotString.append(' ').append(stateLabel);
      }
      dotString.append("\"];\n");

      int actionIndex = 0;
      for (Action action : model.getActions(state)) {
        Object actionLabel = action.label();
        if (action.distribution().size() == 1) {
          IntIterator iterator = action.distribution().support().iterator();
          int successor = iterator.nextInt();
          assert !iterator.hasNext();

          dotString.append(state).append(" -> ").append(successor);
          if (actionLabel != null) {
            dotString.append("[label=\"").append(actionLabel).append("\"]");
          }
          dotString.append(";\n");
        } else {
          String actionNode = String.format("a%d_%d", state, actionIndex);

          dotString.append(state).append(" -> ").append(actionNode)
              .append(" [arrowhead=none,label=\"").append(actionIndex);
          if (actionLabel != null) {
            dotString.append(':').append(actionLabel);
          }

          dotString.append("\" ];\n");
          dotString.append(actionNode).append(" [shape=point,height=0.1];\n");

          action.distribution().forEach((target, probability) -> {
            dotString.append(actionNode).append(" -> ").append(target);
            if (!isEqual(probability, 1.0d)) {
              dotString.append(" [label=\"")
                  .append(String.format("%.3f", probability))
                  .append("\"]");
            }
            dotString.append(";\n");
          });
        }
        actionIndex += 1;
      }
    }
    dotString.append("}\n");
    try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(filename),
        StandardCharsets.UTF_8)) {
      writer.append(dotString.toString());
    } catch (IOException e) {
      throw new AssertionError(e);
    }
  }

  public static boolean isEqual(double d1, double d2) {
    return isZero(d1 - d2);
  }

  public static boolean isOne(double d) {
    return isZero(d - 1.0);
  }

  public static boolean isZero(double d) {
    return Math.abs(d) < MACHINE_EPS;
  }

  public static boolean lessOrEqual(double d1, double d2) {
    return d1 <= d2 || isEqual(d1, d2);
  }
}
