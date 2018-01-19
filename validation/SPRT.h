/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Marco Calignano
    originally taken from Cute Chess (http://github.com/cutechess)
    Copyright (C) 2016 Ilari Pihlajisto

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef SPRT_H
#define SPRT_H
/*!
 * \brief A Sequential Probability Ratio Test
 *
 * The Sprt class implements a Sequential Probability Ratio Test (SPRT) that
 * can be used as a termination criterion for stopping a match between two
 * players when the Elo difference is known to be outside of the specified
 * interval.
 *
 * \sa http://en.wikipedia.org/wiki/Sequential_probability_ratio_test
 */

#include <QMutex>
#include <tuple>

class Sprt
{
	public:
		/*! The result of the test. */
		enum Result
		{
			Continue,	//!< Continue monitoring
			AcceptH0,	//!< Accept null hypothesis H0
			AcceptH1	//!< Accept alternative hypothesis H1
		};

		/*! The result of a chess game. */
		enum GameResult
		{
			NoResult = 0,	//!< Game ended with no result
			Win,		//!< First player won
			Loss,		//!< First player lost
			Draw		//!< Game was drawn
		};

		/*! The status of the test. */
		struct Status
		{
			Result result;	//!< Test result
			double llr;	//!< Log-likelihood ratio
			double lBound;	//!< Lower bound
			double uBound;	//!< Upper bound
		};

		/*! Creates a new uninitialized Sprt object. */
		Sprt();

		/*!
		 * Returns true if the SPRT is uninitialized; otherwise
		 * returns false.
		 */
		bool isNull() const;

		/*!
		 * Initializes the SPRT.
		 *
		 * \a elo0 is the Elo difference between player A and
		 * player B for H0 and \a elo1 for H1.
		 *
		 * \a alpha is the maximum probability for a type I error and
		 * \a beta for a type II error outside interval [elo0, elo1].
		 */
		void initialize(double elo0, double elo1,
				        double alpha, double beta);

		/*! Returns the current status of the test. */
		Status status() const;

		/*! Returns current win/draw/loss score. */
		std::tuple<int, int, int> getWDL() const;

		/*!
		 * Updates the test with \a result.
		 *
		 * After calling this function, status() should be called to
		 * check if H0 or H1 can be accepted.
		 */
		void addGameResult(GameResult result);

	private:
		double m_elo0;
		double m_elo1;
		double m_alpha;
		double m_beta;
		int m_wins;
		int m_losses;
		int m_draws;
		mutable QMutex m_mutex;
};

#endif // SPRT_H
