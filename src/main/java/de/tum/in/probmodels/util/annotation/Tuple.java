package de.tum.in.probmodels.util.annotation;

import org.immutables.value.Value;

@Value.Style(allParameters = true,
             visibility = Value.Style.ImplementationVisibility.PACKAGE,
             typeAbstract = "*",
             typeImmutable = "*Tuple",
             typeImmutableEnclosing = "*Tuple",
             of = "create",
             defaults = @Value.Immutable(builder = false, copy = false))
public @interface Tuple {
}