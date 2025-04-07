import pytest
from importlib.metadata import version, PackageNotFoundError
import re
from packaging import version as packaging_version
import warnings


def read_requirements(file_path='requirements.txt'):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Clean lines, ignore comments, and filter out empty lines
    requirements = []
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace and newlines
        if line and not line.startswith('#'):  # Ensure line is not empty or a comment
            # Match the format 'package==version' and other comparison operators
            match = re.match(r'([a-zA-Z0-9_-]+)(==|>=|<=|<|>|!=)(\d+(\.\d+)*(\.\d+)*)', line)
            if match:
                package, operator, version_str = match.groups()[:3]
                requirements.append((package, operator, version_str))
    return requirements


def test_requirements_versions():
    requirements = read_requirements()
    for package, operator, required_version in requirements:
        try:
            # Get the installed version of the package using importlib.metadata
            installed_version = version(package)

            # Compare versions using packaging.version
            parsed_installed_version = packaging_version.parse(installed_version)
            parsed_required_version = packaging_version.parse(required_version)

            # If the installed version has metadata, issue a warning and accept it
            if parsed_installed_version != parsed_required_version:
                if parsed_installed_version.base_version == parsed_required_version.base_version:
                    warnings.warn(
                        f"Version mismatch for {package}: installed version {installed_version} has additional metadata "
                        f"(e.g., 'post0'), but base version matches expected {required_version}."
                    )
                else:
                    # If versions differ significantly (not just metadata), apply the operator to check if the comparison holds
                    if operator == '==':
                        assert parsed_installed_version == parsed_required_version, \
                            f"{package} version mismatch: expected {required_version}, got {installed_version}"
                    elif operator == '>=':
                        assert parsed_installed_version >= parsed_required_version, \
                            f"{package} version mismatch: expected >= {required_version}, got {installed_version}"
                    elif operator == '<=':
                        assert parsed_installed_version <= parsed_required_version, \
                            f"{package} version mismatch: expected <= {required_version}, got {installed_version}"
                    elif operator == '>':
                        assert parsed_installed_version > parsed_required_version, \
                            f"{package} version mismatch: expected > {required_version}, got {installed_version}"
                    elif operator == '<':
                        assert parsed_installed_version < parsed_required_version, \
                            f"{package} version mismatch: expected < {required_version}, got {installed_version}"
                    elif operator == '!=':
                        assert parsed_installed_version != parsed_required_version, \
                            f"{package} version mismatch: expected != {required_version}, got {installed_version}"

        except PackageNotFoundError:
            pytest.fail(f"{package} is not installed. Run 'pip install -r requirements.txt' from the base folder.")
