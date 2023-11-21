using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Media.TextFormatting;
using System.Diagnostics;
using CNTK;

namespace TestCntkGpu
{
    public class Parall<T>
    {
        private int _threadsNum;
        public int ThreadsNum
        {
            get { return _threadsNum; }
            set { _threadsNum = value; }
        }
        private int _sleepMilliseconds = 10;
        public int SleepMilliseconds
        {
            get { return _sleepMilliseconds; }
            set { _sleepMilliseconds = value; }
        }

        private List<ParallTask<T>> _parallTasksQueue = new List<ParallTask<T>>();

        private List<ParallTask<T>> _completedParallTasks = new List<ParallTask<T>>();
        public List<ParallTask<T>> CompletedParallTasks
        {
            get { return _completedParallTasks; }
            private set { _completedParallTasks = value; }
        }

        public Parall(int threadsNum)
        {
            ThreadsNum = threadsNum;
        }

        public void AddParallTasks(List<ParallTask<T>>  parallTasks)
        {
            _parallTasksQueue.AddRange(parallTasks);
        }

        public void Run()
        {
            CompletedParallTasks.Clear();
            int actualThreadesNum = _parallTasksQueue.Count >= ThreadsNum ? ThreadsNum : _parallTasksQueue.Count;
            ParallTask<T>[] runningParallTasks = new ParallTask<T>[actualThreadesNum];
            int queueIndex = 0;

            for(int i = 0; i < actualThreadesNum; i++)
            {
                ParallTask<T> parallTask = _parallTasksQueue[queueIndex]; //нужно передавать в Task.Run объект, а не массив с индексом объекта, т.к. после увеличения индекса массива в этом потоке, в новом потоке будет выбран объект по новому индексу
                Task.Run(() => parallTask.Run());
                runningParallTasks[i] = _parallTasksQueue[queueIndex];
                queueIndex++;
            }

            bool isComplete = false;
            while(!isComplete)
            {
                Thread.Sleep(SleepMilliseconds);
                isComplete = true;
                for(int i = 0; i < actualThreadesNum; i++)
                {
                    if (runningParallTasks[i].IsComplete)
                    {
                        if(queueIndex < _parallTasksQueue.Count)
                        {
                            ParallTask<T> parallTask = _parallTasksQueue[queueIndex];
                            Task.Run(() => parallTask.Run());
                            runningParallTasks[i] = _parallTasksQueue[queueIndex];
                            queueIndex++;
                            isComplete = false;
                        }
                    }
                    else
                    {
                        isComplete = false;
                    }
                }
                if (isComplete && queueIndex < _parallTasksQueue.Count)
                {
                    isComplete = false;
                }
            }
            CompletedParallTasks.AddRange(_parallTasksQueue);
            _parallTasksQueue.Clear();
        }
    }

    public class ParallTask<T>
    {
        private readonly object locker = new object();
        private bool _isComplete = false;
        public bool IsComplete //реализация потокобезопасного получения и установки свойства
        {
            get
            {
                lock (locker)
                {
                    return _isComplete;
                }
            }
            set
            {
                lock (locker)
                {
                    _isComplete = value;
                }
            }
        }
        private T _res;
        public T Res
        {
            get { return _res; }
            private set { _res = value; }
        }
        private Func<T> _function;
        private Action _action;
        public ParallTask(Func<T> function)
        {
            _function = function;
        }
        public ParallTask(Action action)
        {
            _action = action;
        }
        public void Run()
        {
            if( _function != null )
            {
                _res = _function();
            }
            else
            {
                _action();
            }
            IsComplete = true;
        }
    }
}
