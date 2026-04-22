export default function AboutPage() {
  return (
    <article className="space-y-10 max-w-3xl">
      <header>
        <h1 className="text-3xl font-semibold tracking-tight">How it works</h1>
        <p className="mt-3 text-zinc-400 leading-relaxed">
          Describe a climate in plain English — get real cities that fit, with
          a readable breakdown of <em>why</em>. This page is a short tour of
          what the app does, what the numbers mean, and where it falls short.
        </p>
      </header>

      <Section title="The query format">
        <p>Say something like:</p>
        <Example items={[
          "New York but warmer winters",
          "Tokyo but less humid summers",
          "Seattle but sunnier",
        ]} />
        <p>
          Just a city name works too (&quot;Tokyo&quot; returns climatic twins),
          and so does a description without a starting city (&quot;somewhere
          hot and dry&quot;). The template is <strong>starting city + what
          to change</strong>; either half is optional.
        </p>
      </Section>

      <Section title="What's under the hood">
        <p>
          Every one of the 230 reference cities has a detailed climate{" "}
          <em>fingerprint</em>: monthly averages of 8 weather variables
          (temperature, humidity, rainfall, wind, sunshine, cloud cover,
          pressure, dewpoint) for each month of the year. That's 96 numbers
          per city, built from hourly observations over 2023–2024 from the
          Open-Meteo ERA5 archive.
        </p>
        <p>
          Your query gets converted into a <em>modified fingerprint</em>: start
          from the fingerprint of your anchor city, then shift the numbers
          your query asked about. &quot;Sunnier&quot; bumps up sunshine and
          cloudiness dims. &quot;Warmer winters&quot; only bumps the
          December/January/February temperatures.
        </p>
        <p>
          The app then ranks all 230 cities by how close their fingerprints
          are to your modified target. Dimensions you explicitly asked about
          carry more weight, but every dimension influences the final score.
        </p>
      </Section>

      <Section title="Reading the similarity score">
        <div className="grid sm:grid-cols-2 gap-x-6 gap-y-2">
          <Scale band="80–90%+" toneClass="text-emerald-400" text="very close climatic twin" />
          <Scale band="50–60%" toneClass="text-emerald-300" text="reasonable match" />
          <Scale band="30–40%" toneClass="text-amber-400" text="best given your constraints, but diverges elsewhere" />
          <Scale band="<25%" toneClass="text-zinc-500" text="little in the dataset fits" />
        </div>
        <p>
          The score is deliberately honest. If you ask for an extreme change
          (e.g. &quot;Seattle but much sunnier&quot;), even the top match will
          show a modest percentage — because no real city has Seattle&apos;s
          climate <em>and</em> desert-level sunshine. The ranking still tells
          you which city is closest to your intent; the score tells you how
          close that best match actually is.
        </p>
      </Section>

      <Section title="Reading the Why this matches cards">
        <p>
          Expand any city card in the results to see a list of dots, one per
          weather dimension. They mean:
        </p>
        <ul className="space-y-2 pl-1">
          <li className="flex gap-3">
            <span className="text-emerald-400 mt-0.5">●</span>
            <span>
              <strong className="text-zinc-200">Green — fits.</strong> If you
              asked to change this dimension, the city moved the way you
              asked. Otherwise, the city stays close to your starting city
              here.
            </span>
          </li>
          <li className="flex gap-3">
            <span className="text-zinc-600 mt-0.5">●</span>
            <span>
              <strong className="text-zinc-200">Gray — trade-off.</strong> The
              city is noticeably different from your starting city on this
              dimension — enough to count as a real trade-off, not just normal
              variation between cities. The +/− shows which way it differs,
              not whether that&apos;s good or bad.
            </span>
          </li>
        </ul>
        <p>
          The numbers beside each dot are real averages in real units (°C, %,
          mm/h, km/h) — not LLM-generated prose. That&apos;s the whole point
          of the breakdown: you can sanity-check the ranking against actual
          climate data.
        </p>
      </Section>

      <Section title="The Explorer tab">
        <p>
          The Explorer projects all 230 cities from their 96-number
          fingerprints down to a 2D map so climatically similar cities sit
          close together. The axes are automatically named (e.g. &quot;Summer
          heat and humidity&quot;, &quot;Mild overcast winters&quot;) based on
          which monthly features drive them, so the map is readable without a
          statistics background. Type a query on that page and a red star
          appears where your modified target lands — useful for seeing
          visually which cluster fits your intent.
        </p>
      </Section>

      <Section title="What it doesn't know">
        <ul className="space-y-2 list-disc pl-5 marker:text-zinc-600">
          <li>
            Only the 230 canonical cities are ranked. If you anchor to a city
            outside that list (&quot;Beirut&quot;, &quot;Cedar Rapids&quot;),
            the app fetches its data on demand — expect a ~10-second wait the
            first time that city is ever requested.
          </li>
          <li>
            One modification per query works best. &quot;Warmer winters{" "}
            <em>and</em> drier summers&quot; parses into two separate
            adjustments, both applied at full strength — results may
            over-satisfy one.
          </li>
          <li>
            The dataset covers 2023–2024. It&apos;s current but not a 30-year
            climate normal, so an unusually hot or cold year nudges every
            fingerprint a bit.
          </li>
          <li>
            This is about climate, not weather. It won&apos;t tell you about
            next Tuesday.
          </li>
        </ul>
      </Section>

      <Section title="Credits">
        <p>
          Historical weather data from{" "}
          <a
            href="https://open-meteo.com"
            target="_blank"
            rel="noreferrer"
            className="text-zinc-300 underline decoration-zinc-600 hover:decoration-zinc-300"
          >
            Open-Meteo
          </a>
          . Natural-language parsing via OpenAI with constrained structured
          outputs. Source code at{" "}
          <a
            href="https://github.com/YardenMorad2003/weather-ml-v2"
            target="_blank"
            rel="noreferrer"
            className="text-zinc-300 underline decoration-zinc-600 hover:decoration-zinc-300"
          >
            github.com/YardenMorad2003/weather-ml-v2
          </a>
          .
        </p>
      </Section>
    </article>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-xl border border-zinc-800 bg-zinc-900/30 p-5 space-y-3">
      <h2 className="text-lg font-medium text-zinc-200 tracking-tight">
        {title}
      </h2>
      <div className="text-sm text-zinc-400 leading-relaxed space-y-3">
        {children}
      </div>
    </section>
  );
}

function Example({ items }: { items: string[] }) {
  return (
    <ul className="flex flex-wrap gap-2">
      {items.map((i) => (
        <li
          key={i}
          className="rounded-full border border-zinc-800 bg-zinc-950/60 px-3 py-1 text-xs text-zinc-300 tabular-nums"
        >
          {i}
        </li>
      ))}
    </ul>
  );
}

function Scale({ band, toneClass, text }: { band: string; toneClass: string; text: string }) {
  return (
    <div className="flex items-baseline gap-2">
      <span className={`${toneClass} tabular-nums w-14 text-xs`}>{band}</span>
      <span className="text-xs text-zinc-400">{text}</span>
    </div>
  );
}
