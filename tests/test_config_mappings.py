"""Mapping expression grammar, YAML normalization, and role validation."""

from itertools import product
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest
import yaml
from pydantic import ValidationError

from pyspoc.component.reducers.base import Reducer
from pyspoc.component.statistics.base import Statistic
from pyspoc.config._config_schema import (
    ComponentReference, ComponentStore, MappingRule, MappingsConfig, ReducerSchema,
)
from pyspoc.config._yaml_schemata import LongConfig, ShortConfig


@pytest.mark.parametrize("arrow, statistic_scheme, reducer_scheme, multiple, brackets", list(
    product(
        ["=>", "->"], [False, True], [False, True], [False, True],
        [("", ""), ("[", "]"), ("(", ")"), ("{", "}")],
    )
))
def test_all_single_step_forms(arrow, statistic_scheme, reducer_scheme, multiple, brackets):
    """All eight requested forms accept either arrow and optional group brackets."""
    source = "Statistic@std" if statistic_scheme else "Statistic"
    reducers = ["Reducer@one" if reducer_scheme else "Reducer"]
    if multiple:
        reducers.append("Reducer@two" if reducer_scheme else "OtherReducer")
    targets = ", ".join(reducers)
    targets = f"{brackets[0]}{targets}{brackets[1]}"

    rule = MappingRule.model_validate(f" {source} {arrow} {targets} ")

    assert rule.statistic == ComponentReference(
        name="Statistic", scheme="std" if statistic_scheme else None)
    assert len(rule.reducer_stages) == 1
    assert [ref.name + (f"@{ref.scheme}" if ref.scheme else "")
            for ref in rule.reducer_stages[0]] == reducers


def test_multiple_steps_preserve_order_groups_and_repeated_reducers():
    rule = MappingRule.model_validate(
        "Statistic@std => [Reducer@one, Reducer@two] -> Final => Final@again")

    assert [[(ref.name, ref.scheme) for ref in stage] for stage in rule.reducer_stages] == [
        [("Reducer", "one"), ("Reducer", "two")],
        [("Final", None)],
        [("Final", "again")],
    ]


def test_structured_rules_round_trip_without_losing_stages():
    rule = MappingRule.model_validate("Statistic => [Reducer, OtherReducer] => Final")
    assert MappingRule.model_validate(rule.model_dump()) == rule


@pytest.mark.parametrize("section", ["Positive", "Negative"])
def test_positive_and_negative_share_single_expression_and_list_syntax(section):
    text = "Statistic => Reducer -> Final"
    single = MappingsConfig.model_validate({section: text})
    listed = MappingsConfig.model_validate({section: [text]})
    assert single == listed
    assert len(getattr(single, section)) == 1


@pytest.mark.parametrize("expression", [
    "", "Statistic", "Statistic =>", "=> Reducer", "Statistic => => Reducer",
    "Statistic -> Reducer ->", "[Statistic] => Reducer", "Statistic, Other => Reducer",
    "Statistic => []", "Statistic => [Reducer", "Statistic => Reducer]",
    "Statistic => [[Reducer]]", "Statistic => Reducer,", "Statistic => ,Reducer",
    "Statistic => Reducer,,Final", "Statistic. => Reducer", ".std => Reducer",
    "Statistic => Reducer.", "Statistic => Bad Name", "Statistic ==> Reducer",
    "Statistic => Reducer Final => Other",
    "Statistic => ()", "Statistic => {}", "Statistic => (Reducer]",
    "Statistic => {Reducer)", "Statistic => [Reducer}", "Statistic => (Reducer",
    "Statistic => Reducer}", "Statistic => {(Reducer)}", "Statistic => (Reducer), Final",
    "Statistic => Reducer => (Final}", "Statistic => Reducer.bad(scheme)",
])
def test_malformed_expressions_raise_validation_errors(expression):
    with pytest.raises(ValidationError):
        MappingRule.model_validate(expression)


@pytest.mark.parametrize("value", [
    {"statistic": {"name": "Statistic"}, "reducer_stages": []},
    {"statistic": {"name": "Statistic"}, "reducer_stages": [[]]},
    {"statistic": {"name": ""}, "reducer_stages": [[{"name": "Reducer"}]]},
])
def test_structured_input_cannot_bypass_nonempty_stage_validation(value):
    with pytest.raises(ValidationError):
        MappingRule.model_validate(value)


def test_mixed_group_brackets_keep_identical_semantics():
    mixed = MappingRule.model_validate(
        "Statistic => (Reducer@one, Reducer@two) -> {Final, Final@again} => [Reducer]")
    plain = MappingRule.model_validate(
        "Statistic => Reducer@one, Reducer@two -> Final, Final@again => Reducer")
    assert mixed == plain


def test_curly_groups_preserve_order_and_duplicates_instead_of_becoming_sets():
    rule = MappingRule.model_validate("Statistic => {Reducer@two, Reducer@one, Reducer@two}")
    assert [ref.scheme for ref in rule.reducer_stages[0]] == ["two", "one", "two"]


@pytest.mark.parametrize("scheme", ["bad(name)", "bad{name}", "bad[name]"])
def test_structured_references_also_reject_group_delimiters_in_names(scheme):
    with pytest.raises(ValidationError):
        ComponentReference(name="Reducer", scheme=scheme)


def test_yaml_sequences_and_optional_brackets_normalize_consistently():
    raw = yaml.safe_load("""
Positive:
  - Statistic => Reducer@one, Reducer@two -> Final
Negative:
  - Statistic -> [Reducer@one, Reducer@two] => Final
""")
    mappings = MappingsConfig.model_validate(raw)
    assert mappings.Positive == mappings.Negative


def test_mappings_defaults_are_independent_and_unknown_sections_are_rejected():
    first, second = MappingsConfig(), MappingsConfig()
    first.Positive.append(MappingRule.model_validate("Statistic => Reducer"))
    assert second.Positive == second.Negative == []
    with pytest.raises(ValidationError):
        MappingsConfig.model_validate({"Positve": []})


def test_long_yaml_schema_preserves_mappings_from_example_file():
    path = Path(__file__).resolve().parents[1] / "pyspoc/run_config/test_cfg.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = LongConfig.model_validate(raw)
    assert len(config.Mappings.Positive) == 2
    assert len(config.Mappings.Negative) == 1
    assert config.Mappings.Positive[0].statistic.scheme == "rbf"


def test_short_yaml_schema_preserves_mappings():
    config = ShortConfig.model_validate({
        "Statistics": {}, "ReducedStatistics": {}, "Reducers": {},
        "Mappings": {"Positive": ["Statistic => Reducer => Final"]},
    })
    assert len(config.Mappings.Positive[0].reducer_stages) == 2


@pytest.fixture
def component_config():
    class TestStatistic(Statistic):
        def _compute(self, data):
            return data

    class TestReducer(Reducer):
        def reduce(self, data):
            return data

    # Role validation only needs real instances of the required base classes;
    # no numerical computation or constructor side effects are needed here.
    statistic = object.__new__(TestStatistic)
    reducer = object.__new__(TestReducer)
    return {
        "Statistics": {("Statistic", "std"): statistic},
        "ReducedStatistics": {},
        "Reducers": {
            ("Reducer", "one"): ReducerSchema(ReducerObj=reducer, StatisticFilters=()),
            ("Reducer", "two"): ReducerSchema(ReducerObj=reducer, StatisticFilters=()),
            ("Final", "std"): ReducerSchema(ReducerObj=reducer, StatisticFilters=()),
        },
    }


def test_config_accepts_valid_references_and_unrestricted_schemes(component_config):
    config = ComponentStore.model_validate({
        **component_config,
        "Mappings": {
            "Positive": ["Statistic => [Reducer@one, Reducer@two] => Final"],
            "Negative": "Statistic@std -> Reducer",
        },
    })
    assert len(config.Mappings.Positive[0].reducer_stages) == 2


@pytest.mark.parametrize("section", ["Positive", "Negative"])
@pytest.mark.parametrize("expression, role", [
    ("Missing => Reducer", "Statistic"),
    ("Statistic@missing => Reducer", "Statistic"),
    ("Reducer => Final", "Statistic"),
    ("Statistic => Missing", "Reducer"),
    ("Statistic => Reducer@missing", "Reducer"),
    ("Statistic => Reducer => Statistic", "Reducer"),
    ("Statistic => Reducer => Final@missing", "Reducer"),
])
def test_config_checks_names_schemes_and_roles_at_every_stage(
    component_config, section, expression, role,
):
    with pytest.raises(ValidationError, match=f"unknown configured {role}"):
        ComponentStore.model_validate({
            **component_config, "Mappings": {section: [expression]},
        })


def test_configs_without_mappings_remain_valid(component_config):
    config = ComponentStore.model_validate(component_config)
    assert config.Mappings == MappingsConfig()


@pytest.mark.parametrize("section", ["Positive", "Negative"])
@pytest.mark.parametrize("qualified", list(product([False, True], repeat=3)))
def test_scheme_selection_is_independent_at_every_reducer_stage(
    component_config, section, qualified,
):
    # Mix qualified and unrestricted references at every possible position.
    # None must remain an all-schemes selector, not become the first match.
    names = ["Reducer", "Final", "Reducer"]
    schemes = ["one", "std", "two"]
    stages = [
        f"{name}@{scheme}" if use_scheme else name
        for name, scheme, use_scheme in zip(names, schemes, qualified)
    ]
    expression = "Statistic => " + " -> ".join(stages)
    config = ComponentStore.model_validate({
        **component_config, "Mappings": {section: expression},
    })
    rule = getattr(config.Mappings, section)[0]
    assert [stage[0].name for stage in rule.reducer_stages] == names
    assert [stage[0].scheme for stage in rule.reducer_stages] == [
        scheme if use_scheme else None
        for scheme, use_scheme in zip(schemes, qualified)
    ]


@pytest.mark.parametrize("module", [
    None, "mypackage.mymodule", "D:/mypackage/mymodule", "D:/mypackage/mymodule.py",
    "D:/my package/my.module.py", r"D:\mypackage\my.module.py",
])
@pytest.mark.parametrize("scheme", [None, "standard", "UpperCase.v2"])
def test_qualified_references_with_optional_at_schemes_at_all_positions(module, scheme):
    prefix = f"{module}." if module is not None else ""
    suffix = f"@{scheme}" if scheme is not None else ""
    rule = MappingRule.model_validate(
        f"{prefix}Statistic{suffix} => ({prefix}Reducer{suffix}) -> {prefix}Final{suffix}")

    assert rule.statistic == ComponentReference(name="Statistic", module=module, scheme=scheme)
    assert rule.reducer_stages == [
        [ComponentReference(name="Reducer", module=module, scheme=scheme)],
        [ComponentReference(name="Final", module=module, scheme=scheme)],
    ]
    assert MappingRule.model_validate(rule.model_dump()) == rule


@pytest.mark.parametrize("expression", [
    "Statistic@ => Reducer", "Statistic@@std => Reducer", "Statistic@one@two => Reducer",
    "Statistic => Reducer@", "Statistic => pkg..Reducer@std", "Statistic => .Reducer",
    "Statistic => D:/plugins/module.py.@std", "Statistic => Reducer -> Final@@std",
])
def test_invalid_qualified_references_and_scheme_delimiters(expression):
    with pytest.raises(ValidationError):
        MappingRule.model_validate(expression)


def test_dot_is_only_a_module_delimiter_not_a_scheme_delimiter(component_config):
    rule = MappingRule.model_validate("Statistic.std => Reducer.one")
    assert rule.statistic == ComponentReference(name="std", module="Statistic")
    assert rule.reducer_stages[0][0] == ComponentReference(name="one", module="Reducer")
    with pytest.raises(ValidationError, match="unknown configured Statistic"):
        ComponentStore.model_validate({**component_config, "Mappings": {"Positive": [rule]}})


@pytest.fixture
def duplicate_component_config(component_config):
    # Distinct classes deliberately share short names and scheme names. Fully
    # qualified registry keys allow all entries to coexist without overwriting.
    statistics, reducers = {}, {}
    statistic_base = type(component_config["Statistics"][("Statistic", "std")])
    reducer_base = type(component_config["Reducers"][("Reducer", "one")].ReducerObj)
    paths = {"alpha.plugin": "D:/first dir/my.module.py", "beta.plugin": "D:/second/module.py"}
    for module, path in paths.items():
        stat_type = type("SharedStatistic", (statistic_base,), {"__module__": module})
        reducer_type = type("SharedReducer", (reducer_base,), {"__module__": module})
        metadata = SimpleNamespace(module_reference=module, module_path=path)
        for scheme in ("one", "two"):
            stat = object.__new__(stat_type)
            reducer = object.__new__(reducer_type)
            stat._component_module = metadata
            reducer._component_module = metadata
            statistics[(f"{module}.SharedStatistic", scheme)] = stat
            reducers[(f"{module}.SharedReducer", scheme)] = ReducerSchema(
                ReducerObj=reducer, StatisticFilters=())
    return {"Statistics": statistics, "ReducedStatistics": {}, "Reducers": reducers}


@pytest.mark.parametrize("section", ["Positive", "Negative"])
@pytest.mark.parametrize("schemes", [False, True])
@pytest.mark.parametrize("module", [
    "alpha.plugin", "D:/first dir/my.module.py", "D:/first dir/my.module",
    r"d:\FIRST DIR\my.module.py",
])
def test_module_and_file_qualifiers_disambiguate_duplicate_classes(
    duplicate_component_config, section, schemes, module,
):
    suffix = "@one" if schemes else ""
    expression = (
        f"{module}.SharedStatistic{suffix} => {module}.SharedReducer{suffix}"
        f" -> {module}.SharedReducer@two")
    config = ComponentStore.model_validate({
        **duplicate_component_config, "Mappings": {section: expression},
    })
    rule = getattr(config.Mappings, section)[0]
    assert rule.statistic.scheme == ("one" if schemes else None)
    assert rule.reducer_stages[0][0].module == module
    assert rule.reducer_stages[1][0].scheme == "two"


@pytest.mark.parametrize("expression, role", [
    ("SharedStatistic => alpha.plugin.SharedReducer", "Statistic"),
    ("SharedStatistic@one => alpha.plugin.SharedReducer", "Statistic"),
    ("alpha.plugin.SharedStatistic => SharedReducer", "Reducer"),
    ("alpha.plugin.SharedStatistic => SharedReducer@one", "Reducer"),
    ("alpha.plugin.SharedStatistic => alpha.plugin.SharedReducer -> SharedReducer", "Reducer"),
])
def test_unqualified_duplicate_names_are_rejected(duplicate_component_config, expression, role):
    with pytest.raises(ValidationError, match=f"ambiguous {role}"):
        ComponentStore.model_validate({
            **duplicate_component_config, "Mappings": {"Positive": expression},
        })


def test_qualified_scheme_must_exist_on_the_selected_class(duplicate_component_config):
    with pytest.raises(ValidationError, match="unknown configured Reducer"):
        ComponentStore.model_validate({
            **duplicate_component_config,
            "Mappings": {"Positive": (
                "alpha.plugin.SharedStatistic@one => beta.plugin.SharedReducer@missing")},
        })


def test_qualification_also_works_with_short_registry_keys(component_config):
    statistic = component_config["Statistics"][("Statistic", "std")]
    module = type(statistic).__module__
    config = ComponentStore.model_validate({
        **component_config,
        "Mappings": {"Positive": f"{module}.Statistic@std => {module}.Reducer@one"},
    })
    assert config.Mappings.Positive[0].statistic.module == module


def test_external_file_can_be_identified_without_componentmodule_metadata(
    component_config, monkeypatch,
):
    # External classes may have generated module names. Their loaded module's
    # __file__ must still allow matching a stable source path.
    original = component_config["Statistics"][("Statistic", "std")]
    module = ModuleType("_pyspoc_external_mapping_test")
    module.__file__ = "D:/plugins/statistic.py"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    cls = type("Statistic", (type(original),), {"__module__": module.__name__})
    data = {**component_config, "Statistics": {("Statistic", "std"): object.__new__(cls)}}
    config = ComponentStore.model_validate({
        **data, "Mappings": {"Positive": "D:/plugins/statistic.Statistic@std => Reducer@one"},
    })
    assert config.Mappings.Positive[0].statistic.module == "D:/plugins/statistic"


def test_internal_public_and_implementation_module_qualifiers_match(component_config):
    original = component_config["Statistics"][("Statistic", "std")]
    cls = type("Statistic", (type(original),), {
        "__module__": "pyspoc.component.statistics.example",
    })
    data = {**component_config, "Statistics": {("Statistic", "std"): object.__new__(cls)}}
    ComponentStore.model_validate({
        **data, "Mappings": {"Positive": "pyspoc.statistics.example.Statistic => Reducer"},
    })
