/*
 * semaphore.cpp
 *
 *  Created on: 30-10-2013
 *      Author: Karol Dzitkowski
 */

#include "semaphore.hpp"

namespace ddj
{

Semaphore::Semaphore(unsigned int max)
{
	_max = max;
	_value = 0;
}

Semaphore::~Semaphore(){}

void Semaphore::Wait()
{
	boost::mutex::scoped_lock lock(_mutex);
	for(;_value >= _max;)
	{
		_cond.wait(lock);
	}
	_value++;
}

void Semaphore::Release()
{
	boost::mutex::scoped_lock lock(_mutex);
	if(_value > 0)
	{
		_value--;
		_cond.notify_one();
	}
}

} /* namespace ddj */
