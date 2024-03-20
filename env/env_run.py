import csv


def get_all_task(path):

    # 该函数读取一个CSV文件，该文件包含任务的各种信息，如任务类型、开始时间、结束时间、CPU利用率、内存利用率等。
    type_list = []
    start_time = []
    end_time = []
    cpu_list = []
    mem_list = []

    # 将CSV文件中的每一列数据分别存储在相应的列表（type_list、start_time、end_time、cpu_list、mem_list）中。
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            type_list.append(row[3])
            start_time.append(row[5])
            end_time.append(row[6])
            cpu_list.append(row[7])
            mem_list.append(row[8])

    init_time = int(start_time[0])  # 获取起始时间，将其转换为整数。

    for i in range(len(start_time)):
        # 对之前得到的列表进行一些处理，如将任务类型减一，将时间调整为相对于初始时间的偏移量，将CPU利用率进行百分比转换。
        type_list[i] = int(type_list[i]) - 1
        start_time[i] = int(start_time[i]) - init_time
        end_time[i] = int(end_time[i]) - init_time
        cpu_list[i] = int(cpu_list[i]) / 100.0
        mem_list[i] = float(mem_list[i])
        # 将处理后的列表封装到 all_task 列表中，并返回。
    all_task = [type_list, start_time, end_time, cpu_list, mem_list]

    return all_task


def put_task(task_queue, task):
    # 该函数将一个任务插入到任务队列的开头。
    # 遍历任务队列，将每个任务后移一位，最后将新任务插入到队列的开头。
    # 返回更新后的任务队列

    for i in range(len(task_queue) - 1):
        j = len(task_queue) - i - 1
        task_queue[j] = task_queue[j - 1]
    task_queue[0] = task
    return task_queue


def update_task_queue(master, cur_time, master_id):
    # 该函数用于更新任务队列，包括清理超时的任务和添加新的任务。
    # 清理任务队列中超时的任务，更新任务队列中未完成任务的统计信息。
    # 添加新的任务到任务队列中，这些新任务的开始时间小于当前时间。
    # 对任务队列按照结束时间和开始时间进行排序。
    # 返回更新后的 master 对象。

    # clean task for overtime
    i = 0
    while len(master.task_queue) > i:
        if master.task_queue[i][0] == -1:
            i = i + 1
            continue
        if cur_time >= master.task_queue[i][2]:
            master.undone = master.undone + 1
            master.undone_kind[master.task_queue[i][0]] = master.undone_kind[master.task_queue[i][0]] + 1
            del master.task_queue[i]
        else:
            i = i + 1
    # get new task
    while master.all_task[1][master.all_task_index] < cur_time:
        task = [master.all_task[0][master.all_task_index], master.all_task[1][master.all_task_index],
                master.all_task[2][master.all_task_index], master.all_task[3][master.all_task_index],
                master.all_task[4][master.all_task_index], master_id]
        master.task_queue.append(task)
        master.all_task_index = master.all_task_index + 1

    tmp_list = []
    for i in range(len(master.task_queue)):
        if master.task_queue[i][0] != -1:
            tmp_list.append(master.task_queue[i])
    tmp_list = sorted(tmp_list, key=lambda x: (x[2], x[1]))
    master.task_queue = tmp_list
    return master


def check_queue(task_queue, cur_time):
    # 该函数用于检查任务队列，清理超时的任务。
    # 将队列按照结束时间和开始时间排序。
    # 返回更新后的任务队列，以及未完成任务和相应的任务类型列表。

    task_queue = sorted(task_queue, key=lambda x: (x[2], x[1]))
    undone = [0, 0]
    undone_kind = []
    # clean task for overtime
    i = 0
    while len(task_queue) != i:
        flag = 0
        if cur_time >= task_queue[i][2]:
            undone[task_queue[i][5]] = undone[task_queue[i][5]] + 1
            undone_kind.append(task_queue[i][0])
            del task_queue[i]
            flag = 1
        if flag == 1:
            flag = 0
        else:
            i = i + 1
    return task_queue, undone, undone_kind


def update_docker(node, cur_time, service_coefficient, POD_CPU):
    # 该函数用于更新 Docker 节点的状态，包括执行任务、释放资源等。
    # 执行已经在服务列表中的任务（如果满足条件）。
    # 从任务队列中取出任务并分配资源执行。
    # 更新节点的 CPU 状态，处理已完成和未完成的任务，返回更新后的节点状态以及完成和未完成任务的统计信息

    # 初始化一些变量，用于统计已完成和未完成任务的数量
    done = [0, 0]
    undone = [0, 0]
    done_kind = []
    undone_kind = []

    # 查找在当前时间已完成的任务，并更新节点的状态。
    # find achieved task in current time
    for i in range(len(node.service_list)):
        if node.service_list[i].available_time <= cur_time and len(node.service_list[i].doing_task) > 1:
            done[node.service_list[i].doing_task[5]] = done[node.service_list[i].doing_task[5]] + 1
            done_kind.append(node.service_list[i].doing_task[0])
            node.service_list[i].doing_task = [-1]
            node.service_list[i].available_time = cur_time

    # 执行任务队列中的任务，更新节点的状态。
    # execute task in queue
    i = 0
    while i != len(node.task_queue):
        flag = 0
        for j in range(len(node.service_list)):
            if i == len(node.task_queue):
                break
            if node.task_queue[i][0] == node.service_list[j].kind:
                if node.service_list[j].available_time > cur_time:
                    continue
                if node.service_list[j].available_time <= cur_time:
                    to_do = (node.task_queue[i][3]) / node.service_list[j].cpu
                    if cur_time + to_do <= node.task_queue[i][2] and node.cpu >= POD_CPU * service_coefficient[
                        node.task_queue[i][0]]:
                        node.cpu = node.cpu - POD_CPU * service_coefficient[node.task_queue[i][0]]
                        node.service_list[j].available_time = cur_time + to_do
                        node.service_list[j].doing_task = node.task_queue[i]
                        del node.task_queue[i]
                        flag = 1

                    elif cur_time + to_do > node.task_queue[i][2]:
                        undone[node.task_queue[i][5]] = undone[node.task_queue[i][5]] + 1
                        undone_kind.append(node.task_queue[i][0])
                        del node.task_queue[i]
                        flag = 1
                    elif node.cpu < POD_CPU * service_coefficient[node.task_queue[i][0]]:
                        pass

        if flag == 1:
            flag = 0
        else:
            i = i + 1
    # 返回更新后的节点状态以及已完成和未完成任务的统计信息。
    return node, undone, done, done_kind, undone_kind
