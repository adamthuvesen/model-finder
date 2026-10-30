RUN := uv run --locked --extra models --extra deep

# oauthlib and thrift are held back by databricks-sql-connector, which only the optional
# `databricks` extra installs. Remove these when the connector allows the fixed versions.
REVIEWED_VULNS := --ignore-vuln CVE-2026-49272 --ignore-vuln PYSEC-2026-4114 --ignore-vuln PYSEC-2026-3925 --ignore-vuln PYSEC-2026-3926 --ignore-vuln PYSEC-2026-3927

.PHONY: setup quick check fix slow verify audit lock lint types boundaries rules deps dead test workflows secrets vulnerabilities

# After a clone or a lockfile change. CI runs the same command.
setup:
	uv sync --locked --group dev --extra models --extra deep

# Only the Python files that changed, with no tests. Measured: 0.7s for one file, 2.7s for 49.
quick:
	@$(RUN) python scripts/quick.py

# The fast lane: every offline check, and every test outside tests/verify. CI runs it first.
# The targets run in parallel: 12s, against 32s one after another.
check:
	@$(MAKE) --no-print-directory -j lock lint types boundaries rules deps dead test

# tests/verify holds the slow tests: demos, end-to-end runs, hooks, and generated scripts.
# The coverage gate adds them to the coverage that `make check` recorded in .coverage.
slow:
	@test -f .coverage || { echo "slow: no .coverage from the fast tests. Fix: run make check first" >&2; exit 1; }
	$(RUN) pytest --quiet -n auto tests/verify --cov --cov-append --cov-fail-under=89

# Drive the real CLI offline and keep evidence in .verify/runs/. ARGS: all, list, control, or drive names.
verify:
	@.agents/skills/verify-model-finder/verify $(ARGS)

# Every auto-fixer: lint fixes, then the formatter.
fix:
	$(RUN) ruff check . --fix
	$(RUN) ruff format .

# Needs the network for the vulnerability database.
audit: workflows secrets vulnerabilities

lock:
	uv lock --check

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

types:
	$(RUN) basedpyright --level error
	$(RUN) python scripts/baseline_ratchet.py

boundaries:
	$(RUN) lint-imports

rules:
	$(RUN) ast-grep test --skip-snapshot-tests
	$(RUN) ast-grep scan --error

deps:
	$(RUN) deptry .

dead:
	$(RUN) python scripts/dead_code.py

test:
	$(RUN) pytest --quiet -n auto --ignore=tests/verify --cov --cov-report=

workflows:
	$(RUN) zizmor --offline .github/workflows
	$(RUN) actionlint

secrets:
	gitleaks git --no-banner --redact

vulnerabilities:
	uv export --quiet --frozen --all-extras --all-groups --no-hashes --no-emit-project --output-file .audit-requirements.txt
	$(RUN) pip-audit --requirement .audit-requirements.txt --no-deps --disable-pip --progress-spinner off $(REVIEWED_VULNS); status=$$?; rm -f .audit-requirements.txt; exit $$status
