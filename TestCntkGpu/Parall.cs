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
    public class Parall
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

        private List<ParallTaskBase> _parallTasksQueue = new List<ParallTaskBase>();

        public Parall(int threadsNum)
        {
            _threadsNum = threadsNum;
        }

        public void AddParallTasks(List<ParallTaskBase>  parallTasks)
        {
            _parallTasksQueue.AddRange(parallTasks);
        }

        public void Run()
        {
            int actualThreadesNum = _parallTasksQueue.Count >= _threadsNum ? _threadsNum : _parallTasksQueue.Count;
            ParallTaskBase[] runningParallTasks = new ParallTaskBase[actualThreadesNum];
            int queueIndex = 0;

            for(int i = 0; i < actualThreadesNum; i++)
            {
                ParallTaskBase parallTask = _parallTasksQueue[queueIndex]; //нужно передавать в Task.Run объект, а не массив с индексом объекта, т.к. после увеличения индекса массива в этом потоке, в новом потоке будет выбран объект по новому индексу
                Task.Run(() => parallTask.Run());
                runningParallTasks[i] = _parallTasksQueue[queueIndex];
                queueIndex++;
            }

            bool isComplete = false;
            while(!isComplete)
            {
                Thread.Sleep(_sleepMilliseconds);
                isComplete = true;
                for(int i = 0; i < actualThreadesNum; i++)
                {
                    if (runningParallTasks[i].IsComplete)
                    {
                        if(queueIndex < _parallTasksQueue.Count)
                        {
                            ParallTaskBase parallTask = _parallTasksQueue[queueIndex];
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
            _parallTasksQueue.Clear();
        }
    }
    public abstract class ParallTaskBase
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
        public abstract void Run();
    }
    public class ParallTaskModelEvaluate : ParallTaskBase
    {
        private Function _model;
        private Dictionary<Variable, Value> _inputDataMap;
        private Dictionary<Variable, Value> _outputDataMap;
        private DeviceDescriptor _modelDevice;
        public ParallTaskModelEvaluate(Function model, Dictionary<Variable, Value> inputDataMap, Dictionary<Variable, Value> outputDataMap, DeviceDescriptor modelDevice)
        {
            _model = model;
            _inputDataMap = inputDataMap;
            _outputDataMap = outputDataMap;
            _modelDevice = modelDevice;
        }
        public override void Run()
        {
            _model.Evaluate(_inputDataMap, _outputDataMap, _modelDevice);
            IsComplete = true;
        }
    }
}
