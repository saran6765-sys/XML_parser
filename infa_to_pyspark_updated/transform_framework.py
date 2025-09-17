"""
Lightweight transformation framework for Databricks PySpark jobs.

Provides modular helpers for Expressions, Filters, Joins, Lookups, Routers,
Aggregations, Sequence generation, Delta MERGE, and Logging/Validation.

Intended for use inside Databricks; functions reference pyspark APIs.
"""
from typing import Dict, List, Optional, Tuple
import json


def setup_spark(spark, shuffle_partitions: int = 16, enable_aqe: bool = True):
    spark.conf.set("spark.sql.shuffle.partitions", str(shuffle_partitions))
    if enable_aqe:
        spark.conf.set("spark.sql.adaptive.enabled", "true")
        spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
        spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")


def log_schema_and_count(df, name: str):
    print(f"== {name} schema ==")
    df.printSchema()
    print(f"== {name} count == {df.count()}")


def apply_expression(df, expr_specs: List[Dict]):
    """expr_specs: list of {name, expr} for withColumn expressions."""
    from pyspark.sql.functions import expr

    out = df
    for spec in expr_specs or []:
        col_name = spec.get("name")
        e = spec.get("expr")
        if col_name and e:
            out = out.withColumn(col_name, expr(e))
    return out


def apply_filter(df, condition: str):
    return df.filter(condition) if condition else df


def apply_join(left, right, on: List[str], how: str = "inner", broadcast_small: bool = False):
    from pyspark.sql.functions import broadcast

    r = broadcast(right) if broadcast_small else right
    return left.join(r, on=on, how=how)


def apply_lookup_range(
    left,
    right,
    key: str,
    min_col: str,
    max_col: str,
    value_cols: Optional[List[str]] = None,
    include_min: bool = True,
    include_max: bool = False,
):
    """Lookup by range with configurable bounds.

    Condition: (key >= min) and (key < max) by default.
    Set include_max=True to use <= max.
    """
    from pyspark.sql.functions import col

    lower = col(key) >= col(min_col) if include_min else col(key) > col(min_col)
    upper = col(key) <= col(max_col) if include_max else col(key) < col(max_col)
    cond = lower & upper
    join_df = left.join(right, cond, how="left")
    if value_cols:
        keep = [c for c in left.columns] + value_cols
        join_df = join_df.select(*keep)
    return join_df


def lookup_asof(
    left,
    right,
    partition_keys: List[str],
    left_time_col: str,
    right_time_col: str,
    direction: str = "prior",
    value_cols: Optional[List[str]] = None,
):
    """AS-OF lookup: join nearest prior/next right record for each left row.

    - direction: "prior" (right_time <= left_time, pick latest) or "next"
      (right_time >= left_time, pick earliest)
    - Returns left columns + selected right value_cols (if provided) or all right columns.
    """
    from pyspark.sql.functions import col, monotonically_increasing_id, row_number
    from pyspark.sql.window import Window

    left_id = left.withColumn("_left_row_id", monotonically_increasing_id())
    on_eq = [left_id[k] == right[k] for k in partition_keys]
    if direction == "next":
        cond = (col(right_time_col) >= left_id[left_time_col])
        order = [col(right_time_col).asc()]
    else:
        cond = (col(right_time_col) <= left_id[left_time_col])
        order = [col(right_time_col).desc()]

    joined = left_id.join(right, on=on_eq + [cond], how="left")
    w = Window.partitionBy("_left_row_id").orderBy(*order)
    ranked = joined.withColumn("_asof_rn", row_number().over(w))
    best = ranked.filter(col("_asof_rn") == 1)

    if value_cols:
        keep = [c for c in left.columns] + value_cols
        out = best.select(*keep)
    else:
        out = best.select(*left.columns, *[c for c in best.columns if c not in left.columns and not c.startswith("_")])
    return out


def apply_router(df, rules: List[Tuple[str, str]]):
    """rules: list of (name, condition) -> returns dict[name] = filtered df."""
    out = {}
    for name, cond in rules or []:
        out[name] = df.filter(cond)
    return out


def apply_aggregations(df, group_cols: List[str], agg_exprs: Dict[str, str]):
    """agg_exprs: mapping new_col -> SQL expression (e.g., 'sum(gross_sal)')."""
    from pyspark.sql.functions import expr

    grouped = df.groupBy(*group_cols) if group_cols else df
    if not agg_exprs:
        return grouped
    agg_cols = [expr(f"{e} as {k}") for k, e in agg_exprs.items()]
    return grouped.agg(*agg_cols)


def apply_rank(
    df,
    rank_col: str = "rank",
    partition_by: Optional[List[str]] = None,
    order_by: Optional[List[str]] = None,
    method: str = "row_number",
):
    """Add a rank column using window functions.

    method: 'row_number' (default), 'rank', or 'dense_rank'
    """
    from pyspark.sql.window import Window
    from pyspark.sql.functions import row_number, rank, dense_rank, col

    w = Window.partitionBy(*partition_by) if partition_by else Window.partitionBy()
    if order_by:
        w = w.orderBy(*[col(c) for c in order_by])
    fn = row_number if method == "row_number" else (rank if method == "rank" else dense_rank)
    return df.withColumn(rank_col, fn().over(w))


def apply_sorter(df, order_by: List[Tuple[str, str]]):
    """Sort DataFrame by a list of (column, direction) where direction is 'asc' or 'desc'."""
    from pyspark.sql.functions import col

    exprs = [getattr(col(c), d.lower())() if d else col(c).asc() for c, d in (order_by or [])]
    return df.orderBy(*exprs) if exprs else df


def apply_normalizer(
    df,
    trims: Optional[List[str]] = None,
    lower: Optional[List[str]] = None,
    upper: Optional[List[str]] = None,
    default_map: Optional[Dict[str, str]] = None,
):
    """Normalize text columns: trim/lower/upper and apply defaults via coalesce.

    default_map: {column -> default_literal_sql}
    """
    from pyspark.sql.functions import trim, lower as f_lower, upper as f_upper, coalesce, expr, col

    out = df
    for c in (trims or []):
        if c in out.columns:
            out = out.withColumn(c, trim(col(c)))
    for c in (lower or []):
        if c in out.columns:
            out = out.withColumn(c, f_lower(col(c)))
    for c in (upper or []):
        if c in out.columns:
            out = out.withColumn(c, f_upper(col(c)))
    for c, d in (default_map or {}).items():
        if c in out.columns:
            out = out.withColumn(c, coalesce(col(c), expr(d)))
    return out


def add_sequence(df, col_name: str, method: str = "row_number", partition_by: Optional[List[str]] = None, order_by: Optional[List[str]] = None):
    from pyspark.sql.window import Window
    from pyspark.sql.functions import row_number, monotonically_increasing_id, col

    if method == "monotonically_increasing_id":
        return df.withColumn(col_name, monotonically_increasing_id())

    # default: row_number()
    w = Window.partitionBy(*partition_by) if partition_by else Window.partitionBy()
    if order_by:
        w = w.orderBy(*[col(c) for c in order_by])
    return df.withColumn(col_name, row_number().over(w))


def delta_merge(spark, df, target_table: str, on: str, update_set: Dict[str, str], insert_set: Optional[Dict[str, str]] = None, when_not_matched_insert: bool = True):
    """Execute a Delta MERGE INTO using a temp view for df.

    on: SQL condition string for merge keys (e.g., 't.emp_id = s.emp_id').
    update_set/insert_set: dict of target_col -> source_expression.
    """
    view_name = "_merge_src"
    df.createOrReplaceTempView(view_name)
    tgt_alias = "t"
    src_alias = "s"

    upd = ",\n        ".join([f"{tgt_alias}.{k} = {v}" for k, v in (update_set or {}).items()])
    ins_cols = ", ".join((insert_set or update_set or {}).keys())
    ins_vals = ", ".join((insert_set or update_set or {}).values())

    sql = f"""
    MERGE INTO {target_table} {tgt_alias}
    USING {view_name} {src_alias}
    ON {on}
    WHEN MATCHED THEN UPDATE SET
        {upd}
    {f'WHEN NOT MATCHED THEN INSERT ({ins_cols}) VALUES ({ins_vals})' if when_not_matched_insert else ''}
    """
    spark.sql(sql)


def safe_delta_merge(
    spark,
    df,
    target_table: str,
    on: str,
    update_set: Dict[str, str],
    insert_set: Optional[Dict[str, str]] = None,
    when_not_matched_insert: bool = True,
    overwrite_fallback: bool = False,
    repartition_before_write: Optional[int] = None,
):
    """MERGE with basic error handling and optional overwrite fallback.

    Returns (ok: bool, message: str)
    """
    try:
        delta_merge(
            spark,
            df,
            target_table,
            on,
            update_set,
            insert_set=insert_set,
            when_not_matched_insert=when_not_matched_insert,
        )
        return True, "merge_ok"
    except Exception as e:  # noqa: BLE001
        msg = f"merge_failed: {e}"
        if overwrite_fallback:
            try:
                wdf = df
                if repartition_before_write:
                    wdf = wdf.repartition(int(repartition_before_write))
                wdf.write.mode("overwrite").format("delta").saveAsTable(target_table)
                return True, msg + "; overwrite_fallback_ok"
            except Exception as e2:  # noqa: BLE001
                return False, msg + f"; overwrite_fallback_failed: {e2}"
    return False, msg


def allocate_sequence_range(
    spark,
    sequence_table: str,
    step: int = 1,
    retry: int = 3,
) -> int:
    """Allocate a global sequence range atomically using Delta optimistic concurrency.

    Returns the starting value allocated (inclusive). The allocated range is
    [start, start+step-1]. Creates the table on first use.
    Table schema: (id INT, next_value BIGINT)
    """
    from delta.tables import DeltaTable  # type: ignore

    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {sequence_table} (id INT, next_value BIGINT)
        USING DELTA
        TBLPROPERTIES (delta.feature.allowColumnDefaults = true)
        """
    )
    if spark.table(sequence_table).where("id = 1").count() == 0:
        spark.sql(f"INSERT INTO {sequence_table} VALUES (1, 1)")

    start = None
    for _ in range(max(1, retry)):
        curr = spark.table(sequence_table).where("id = 1").select("next_value").limit(1).collect()[0][0]
        new_val = int(curr) + int(step)
        dt = DeltaTable.forName(spark, sequence_table)
        # Optimistic update: only succeed if next_value is unchanged
        dt.update(
            condition=f"id = 1 AND next_value = {curr}",
            set={"next_value": f"{new_val}"},
        )
        # Verify
        after = spark.table(sequence_table).where("id = 1").select("next_value").limit(1).collect()[0][0]
        if int(after) == new_val:
            start = int(curr)
            break
    if start is None:
        raise RuntimeError("allocate_sequence_range failed after retries")
    return start


def assign_sequence_from_start(
    df,
    col_name: str,
    start_value: int,
    order_by: Optional[List[str]] = None,
    partition_by: Optional[List[str]] = None,
):
    """Assign a monotonically increasing sequence based on a starting value.

    Uses row_number over the specified window and offsets by start_value-1.
    Combine with allocate_sequence_range() for global SK allocation.
    """
    from pyspark.sql.window import Window
    from pyspark.sql.functions import row_number, col, expr

    w = Window.partitionBy(*partition_by) if partition_by else Window.partitionBy()
    if order_by:
        w = w.orderBy(*[col(c) for c in order_by])
    return df.withColumn(col_name, expr(str(start_value - 1)) + row_number().over(w))


def write_audit_log(
    spark,
    table: str,
    run_id: str,
    status: str,
    metrics: Optional[Dict[str, int]] = None,
    issues: Optional[List[str]] = None,
):
    """Append a single audit row with run metadata and validation issues.

    Schema: (ts TIMESTAMP, run_id STRING, status STRING, metrics_json STRING, issues_json STRING)
    """
    metrics_json = json.dumps(metrics or {}, ensure_ascii=False)
    issues_json = json.dumps(issues or [], ensure_ascii=False)
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
          ts TIMESTAMP,
          run_id STRING,
          status STRING,
          metrics_json STRING,
          issues_json STRING
        ) USING DELTA
        """
    )
    spark.sql(
        f"INSERT INTO {table} VALUES (current_timestamp(), '{run_id}', '{status}', '{metrics_json}', '{issues_json}')"
    )


def add_delta_check_constraints(
    spark,
    table: str,
    constraints: Dict[str, str],
):
    """Add CHECK constraints (if not exists) to a Delta table.
    constraints: mapping name -> SQL predicate
    """
    for name, pred in (constraints or {}).items():
        try:
            spark.sql(f"ALTER TABLE {table} ADD CONSTRAINT IF NOT EXISTS {name} CHECK ({pred})")
        except Exception:
            # Older runtimes may not support IF NOT EXISTS; try without
            try:
                spark.sql(f"ALTER TABLE {table} ADD CONSTRAINT {name} CHECK ({pred})")
            except Exception:
                pass


def validate_table_schema_exact(
    spark,
    table: str,
    expected_fields: List[Tuple[str, str]],
):
    """Validate that table schema matches expected (name + dataType string).
    Returns list of issues (empty if ok).
    """
    issues: List[str] = []
    try:
        schema = spark.table(table).schema
    except Exception as e:  # noqa: BLE001
        return [f"cannot_read_table: {e}"]
    actual = [(f.name.lower(), f.dataType.simpleString().lower()) for f in schema.fields]
    exp = [(n.lower(), (t or "").lower()) for (n, t) in expected_fields]
    if len(actual) != len(exp):
        issues.append(f"field_count_mismatch: actual={len(actual)} expected={len(exp)}")
    for idx, (en, et) in enumerate(exp):
        if idx >= len(actual):
            issues.append(f"missing_actual_field_at_{idx}: {en} {et}")
            continue
        an, at = actual[idx]
        if an != en:
            issues.append(f"name_mismatch_at_{idx}: actual={an} expected={en}")
        if et and at != et:
            issues.append(f"type_mismatch_{en}: actual={at} expected={et}")
    return issues


def validate_partitioning(spark, table: str, expected_cols: List[str]):
    """Check Delta table partitioning using DESCRIBE DETAIL.
    Returns list of issues.
    """
    try:
        detail = spark.sql(f"DESCRIBE DETAIL {table}").collect()[0]
        parts = [c.lower() for c in detail[detail.fieldIndex("partitionColumns")]]
    except Exception as e:  # noqa: BLE001
        return [f"describe_detail_failed: {e}"]
    exp = [c.lower() for c in expected_cols or []]
    missing = [c for c in exp if c not in parts]
    return [f"missing_partition_cols: {', '.join(missing)}"] if missing else []


def enforce_schema_df(df, expected_fields: List[Tuple[str, str]]):
    """Cast and select columns to enforce exact target schema ordering/types.
    expected_fields: list of (name, type)
    """
    from pyspark.sql.functions import col, lit, expr

    cols = [name for name, _ in expected_fields]
    casted = df
    for name, typ in expected_fields:
        if name in df.columns:
            casted = casted.withColumn(name, expr(f"CAST({name} AS {typ})") if typ else col(name))
        else:
            casted = casted.withColumn(name, lit(None).cast(typ) if typ else lit(None))
    # select in order
    return casted.select(*cols)


def validate_not_null_df(df, cols: List[str]) -> List[str]:
    issues: List[str] = []
    for c in cols or []:
        try:
            cnt = df.filter(f"{c} IS NULL").limit(1).count()
        except Exception as e:  # noqa: BLE001
            issues.append(f"not_null_check_failed_{c}: {e}")
            continue
        if cnt > 0:
            issues.append(f"null_values_in_{c}")
    return issues


def validate_unique_df(df, cols: List[str]) -> List[str]:
    """Check duplicates for a composite key.
    Returns issues list.
    """
    issues: List[str] = []
    if not cols:
        return issues
    try:
        dup_cnt = df.groupBy(*cols).count().filter("count > 1").limit(1).count()
        if dup_cnt > 0:
            issues.append(f"duplicate_keys_in_{'_'.join(cols)}")
    except Exception as e:  # noqa: BLE001
        issues.append(f"unique_check_failed_{'_'.join(cols)}: {e}")
    return issues


def validate_surrogate_key_df(df, sk_col: str) -> List[str]:
    issues: List[str] = []
    try:
        nulls = df.filter(f"{sk_col} IS NULL").limit(1).count()
        if nulls > 0:
            issues.append(f"surrogate_key_nulls_in_{sk_col}")
        dup = df.groupBy(sk_col).count().filter("count > 1").limit(1).count()
        if dup > 0:
            issues.append(f"surrogate_key_duplicates_in_{sk_col}")
    except Exception as e:  # noqa: BLE001
        issues.append(f"surrogate_key_validation_failed_{sk_col}: {e}")
    return issues
