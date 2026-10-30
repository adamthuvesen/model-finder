RUN := uv run --locked --extra models --extra deep

# oauthlib and thrift are held back by databricks-sql-connector, which only the optional
# `databricks` extra installs. Remove these when the connector allows the fixed versions.
REVIEWED_VULNS := --ignore-vuln CVE-2026-49272 --ignore-vuln PYSEC-2026-4114 --ignore-vuln PYSEC-2026-3925 --ignore-vuln PYSEC-2026-3926 --ignore-vuln PYSEC-2026-3927

.PHONY: quick fast verify check audit lock lint types boundaries rules deps dead test workflows secrets vulnerabilities

# Only the Python files that changed, with no tests. Measured: 0.7s for one file, 2.7s for 49.
quick:
	@.venv/bin/python scripts/quick.py

# The fast lane: every offline check, and every test outside tests/verify. CI runs it first.
# The targets run in parallel: 12s, against 32s one after another.
fast:
	@$(MAKE) --no-print-directory -j lock lint types boundaries rules deps dead test

# tests/verify holds the slow tests: demos, end-to-end runs, hooks, and generated scripts.
# The coverage gate needs the whole suite, so this runs every test.
verify:
	$(RUN) pytest --quiet -n auto --cov --cov-fail-under=89

# The full offline gate, as CI runs it.
check: fast verify

# Needs the network for the vulnerability database.
audit: workflows secrets vulnerabilities

lock:
	uv lock --check

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

types:
	$(RUN) basedpyright --level error

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
	$(RUN) pytest --quiet -n auto --ignore=tests/verify

workflows:
	$(RUN) zizmor --offline .github/workflows
	$(RUN) actionlint

secrets:
	gitleaks git --no-banner --redact

vulnerabilities:
	uv export --quiet --frozen --all-extras --all-groups --no-hashes --no-emit-project --output-file .audit-requirements.txt
	$(RUN) pip-audit --requirement .audit-requirements.txt --no-deps --disable-pip --progress-spinner off $(REVIEWED_VULNS); status=$$?; rm -f .audit-requirements.txt; exit $$status
