class Cloud:
    def __init__(self, task_queue, service_list, cpu, mem):
        # task_queue: 一个任务队列，用于存储云环境中待执行的任务。
        # service_list: 一个服务列表，存储当前云环境中运行的服务。
        # cpu: 云环境的CPU容量（GHz）。
        # mem: 云环境的内存容量（GB）。

        self.task_queue = task_queue
        self.service_list = service_list
        self.cpu = cpu  # GHz
        self.mem = mem  # GB


class Node:
    def __init__(self, cpu, mem, service_list, task_queue):
        # 该类表示一个计算节点，具有以下属性：
        # cpu: 节点的CPU容量（GHz）。
        # cpu_max: 节点的最大CPU容量。
        # mem: 节点的内存容量（MB）。
        # mem_max: 节点的最大内存容量。
        # service_list: 一个服务列表，存储当前节点上运行的服务。
        # task_queue: 一个任务队列，存储当前节点中待执行的任务。

        self.cpu = cpu
        self.cpu_max = cpu
        self.mem = mem
        self.mem_max = mem
        self.service_list = service_list
        self.task_queue = task_queue


class Master:
    def __init__(self, cpu, mem, node_list, task_queue, all_task, all_task_index, done, undone, done_kind, undone_kind):
        # 该类表示主控制器，负责协调整个系统，具有以下属性：
        # cpu: 主控制器的CPU容量（GHz）。
        # mem: 主控制器的内存容量（MB）。
        # node_list: 一个节点列表，存储所有计算节点。
        # task_queue: 一个任务队列，用于存储主控制器中待执行的任务。
        # all_task: 存储所有任务的信息的列表。
        # all_task_index: 一个索引，表示在 all_task 中的当前位置。
        # done: 一个已完成任务的统计列表。
        # undone: 一个未完成任务的统计列表。
        # done_kind: 已完成任务的类型列表。
        # undone_kind: 未完成任务的类型列表。

        self.cpu = cpu  # GHz
        self.mem = mem  # MB
        self.node_list = node_list
        self.task_queue = task_queue
        self.all_task = all_task
        self.all_task_index = all_task_index
        self.done = done
        self.undone = undone
        self.done_kind = done_kind
        self.undone_kind = undone_kind


class Docker:
    def __init__(self, mem, cpu, available_time, kind, doing_task):
        # 该类表示Docker容器，用于模拟容器化的任务执行，具有以下属性：
        # mem: Docker容器的内存需求。
        # cpu: Docker容器的CPU需求。
        # available_time: Docker容器的可用时间。
        # kind: Docker容器的类型。
        # doing_task: 当前正在执行的任务。

        self.mem = mem
        self.cpu = cpu
        self.available_time = available_time
        self.kind = kind
        self.doing_task = doing_task
