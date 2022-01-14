import os


def path_to_save_results() -> str:
    path = project_path()
    save_path = os.path.join(path, 'results_of_experiments')
    return save_path


def project_path() -> str:
    name_project = 'automl-crash-test'
    abs_path = os.path.abspath(os.path.curdir)
    while os.path.basename(abs_path) != name_project:
        abs_path = os.path.dirname(abs_path)
    return abs_path


def ExceptionDecorator(function_to_decorate):
    def Exception_wrapper():
        try:
            function_to_decorate()
        except Exception:
            return None

        return Exception_wrapper
